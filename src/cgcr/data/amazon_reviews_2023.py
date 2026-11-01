"""stream Amazon gzip records, resolve variants, and retain filtered parquet."""

from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time

import pyarrow as pa
import pyarrow.parquet as pq

CATALOG_FIELDS = ["title", "features", "details", "price", "average_rating",
                  "rating_number", "categories", "store"]
CATALOG_SCHEMA = pa.schema([
    ("parent_asin", pa.string()), ("title", pa.string()),
    ("features", pa.list_(pa.string())), ("details", pa.string()),
    ("price", pa.string()), ("average_rating", pa.float64()),
    ("rating_number", pa.int64()), ("categories", pa.list_(pa.string())),
    ("store", pa.string()),
])
REVIEW_SCHEMA = pa.schema([
    ("rating", pa.float64()), ("title", pa.string()), ("text", pa.string()),
    ("timestamp", pa.int64()), ("verified_purchase", pa.bool_()),
    ("helpful_vote", pa.int64()), ("user_id", pa.string()),
])


def iter_filtered(path, ids, field="parent_asin", progress=None):
    """Read gzip line by line; decode JSON only for matching identifier tokens.

    Identifiers are taken from the top-level record after a cheap byte prefilter.
    The regex accepts JSON whitespace and escaped identifiers. No whole-file read.
    Nonmatching rows need no expensive JSON parse. All lines are still decompressed.
    """
    wanted = set(ids)
    wanted_bytes = {value.encode("utf-8") for value in wanted}
    pattern = re.compile(rb'"' + field.encode() + rb'"\s*:\s*"((?:[^"\\]|\\.)*)"')
    started = last = time.monotonic()
    matched = 0
    with gzip.open(path, "rb") as source:
        for count, line in enumerate(source, 1):
            token = pattern.search(line)
            if token:
                raw = token.group(1)
                if raw in wanted_bytes or b"\\" in raw:
                    row = json.loads(line)
                    if row.get(field) in wanted:
                        matched += 1
                        yield row
            if progress and count % 100000 == 0 and time.monotonic() - last >= 15:
                progress(f"{Path(path).name}: {field} scan {count:,} rows, {matched:,} matches, "
                         f"{time.monotonic() - started:.1f}s")
                last = time.monotonic()


def child_parent_mapping(path, asins, output_dir, progress=None):
    """Resolve only requested child ASINs; exclude ambiguous child/parent pairs."""
    found, conflicts = {}, set()
    for row in iter_filtered(path, asins, "asin", progress):
        child, parent = row["asin"], row.get("parent_asin")
        if not parent or child in conflicts:
            continue
        if child in found and found[child] != parent:
            conflicts.add(child)
            found.pop(child)
        else:
            found[child] = parent
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([("asin", pa.string()), ("parent_asin", pa.string())])
    pq.write_table(pa.Table.from_pylist(
        [{"asin": k, "parent_asin": v} for k, v in sorted(found.items())], schema=schema),
        out / "asin_parent_map.parquet")
    return found, sorted(conflicts)


def _strings(value):
    if value is None:
        return []
    return [str(v) for v in value] if isinstance(value, list) else [str(value)]


def filter_metadata(path, asins, output_dir, progress=None):
    """Retain selected parent catalog rows; details stay lossless JSON text.

    Dynamic details dictionaries cannot share a stable Arrow struct across categories.
    Price is nullable source text, avoiding lossy conversions of strings/currencies.
    """
    catalog = {}
    for row in iter_filtered(path, asins, progress=progress):
        parent = row["parent_asin"]
        catalog[parent] = {
            "parent_asin": parent, "title": str(row.get("title") or ""),
            "features": _strings(row.get("features")),
            "details": json.dumps(row.get("details") or {}, ensure_ascii=False, sort_keys=True),
            "price": None if row.get("price") is None else str(row["price"]),
            "average_rating": row.get("average_rating"),
            "rating_number": row.get("rating_number"),
            "categories": _strings(row.get("categories")),
            "store": None if row.get("store") is None else str(row["store"]),
        }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(list(catalog.values()), schema=CATALOG_SCHEMA),
                   out / "metadata.parquet")
    return catalog


def hash_user_id(value, salt):
    """Use a configured fixed salt; preserve missing IDs as null."""
    if value is None:
        return None
    return hashlib.sha256((salt + "\0" + str(value)).encode()).hexdigest()


def filter_reviews(path, parents, output_dir, salt, progress=None):
    """Stream all variant reviews of selected parents into per-item parquet.

    A temporary SQLite spool bounds memory and open file descriptors. User IDs are
    hashed before entering the spool. A second gzip pass is necessary: children
    discovered late in the mapping pass can have parent reviews earlier in the file.
    """
    out = Path(output_dir)
    reviews_dir = out / "reviews"
    reviews_dir.mkdir(parents=True, exist_ok=True)
    spool = out / "reviews_spool.sqlite"
    spool.unlink(missing_ok=True)
    db = sqlite3.connect(spool)
    counts = Counter({parent: 0 for parent in parents})
    try:
        db.execute("PRAGMA journal_mode=OFF")
        db.execute("PRAGMA synchronous=OFF")
        db.execute("CREATE TABLE reviews (parent TEXT, payload TEXT)")
        pending = []
        for row in iter_filtered(path, parents, progress=progress):
            parent = row["parent_asin"]
            review = {key: row.get(key) for key in REVIEW_SCHEMA.names}
            review["user_id"] = hash_user_id(row.get("user_id"), salt)
            pending.append((parent, json.dumps(review, ensure_ascii=False)))
            counts[parent] += 1
            if len(pending) >= 5000:
                db.executemany("INSERT INTO reviews VALUES (?, ?)", pending)
                db.commit()
                pending.clear()
        if pending:
            db.executemany("INSERT INTO reviews VALUES (?, ?)", pending)
            db.commit()
        db.execute("CREATE INDEX parent_idx ON reviews(parent)")
        for index, parent in enumerate(sorted(parents)):
            if not re.fullmatch(r"[A-Za-z0-9_-]+", parent):
                raise ValueError(f"Unsafe ASIN filename: {parent!r}")
            with pq.ParquetWriter(reviews_dir / f"{parent}.parquet", REVIEW_SCHEMA) as writer:
                cursor = db.execute("SELECT payload FROM reviews WHERE parent=? ORDER BY rowid", (parent,))
                while chunk := cursor.fetchmany(1000):
                    writer.write_table(pa.Table.from_pylist(
                        [json.loads(r[0]) for r in chunk], schema=REVIEW_SCHEMA))
            if progress and index % 1000 == 0:
                progress(f"Review parquet: {index + 1:,}/{len(parents):,} items")
    finally:
        db.close()
        spool.unlink(missing_ok=True)
    return dict(counts)
