"""budgeted downloads, streaming join study, and reproducible pilot preparation."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import sys
import time
from urllib.request import Request, urlopen

import pyarrow as pa
import pyarrow.parquet as pq

from cgcr.data.amazon_reviews_2023 import (
    CATALOG_SCHEMA, child_parent_mapping, filter_metadata, filter_reviews,
)
from cgcr.data.candidate_sets import coverage, choose_category, build_candidates, split_queries
from cgcr.data.esci import load_esci, read_filtered
from cgcr.utils.io import read_yaml, write_yaml, write_jsonl
from cgcr.utils.logging import start_run, get_logger

CATEGORIES = ["Electronics", "Home_and_Kitchen",
              "Tools_and_Home_Improvement"]
ROOT = Path(__file__).resolve().parents[1]
LIMIT = 60_000_000_000


def fetch_ranges(item, logger, root=ROOT, workers=16, chunk_size=8 * 1024 * 1024):
    """Bounded concurrent HTTP ranges, persisted completion map, no whole-file RAM.

    Range responses must identify the exact requested byte interval. Completed
    ranges survive interruption; only the incomplete chunks are fetched again.
    """
    path = root / item["path"]
    partial = path.with_suffix(path.suffix + ".ranges.part")
    state_path = path.with_suffix(path.suffix + ".ranges.json")
    length = item["content_length"]
    signature = {"url": item["url"], "length": length, "chunk_size": chunk_size}
    sequential_partial = path.with_suffix(path.suffix + ".part")
    if not partial.exists() and sequential_partial.exists():
        prefix = sequential_partial.stat().st_size
        sequential_partial.replace(partial)
        save_json(state_path, signature | {"completed": list(range(prefix // chunk_size))})
    state = json.loads(state_path.read_text()) if state_path.exists() else signature | {"completed": []}
    if any(state[k] != v for k, v in signature.items()):
        raise RuntimeError("Download range state differs from plan")
    completed = set(state["completed"]) if partial.exists() else set()
    segments = [(index, start, min(start + chunk_size, length) - 1)
                for index, start in enumerate(range(0, length, chunk_size))]
    already = sum(end - start + 1 for index, start, end in segments if index in completed)
    if shutil.disk_usage(root).free < length - already + 2_000_000_000:
        raise RuntimeError(f"Insufficient free space for range download {path}")
    fd = os.open(partial, os.O_CREAT | os.O_RDWR, 0o600)
    os.ftruncate(fd, length)

    def transfer(segment):
        index, start, end = segment
        # Smaller HTTP requests limit loss when the source closes a response.
        # Keep 8-MiB checkpoint geometry so older completed chunks stay reusable.
        for sub_start in range(start, end + 1, 2 * 1024 * 1024):
            sub_end = min(sub_start + 2 * 1024 * 1024 - 1, end)
            cursor, failures = sub_start, 0
            while cursor <= sub_end:
                request_start = cursor
                try:
                    request = Request(item["url"], headers={"Range": f"bytes={cursor}-{sub_end}",
                                      "User-Agent": "CGCR-T2-data-pipeline/1.0"})
                    with urlopen(request, timeout=90) as response:
                        expected = f"bytes {cursor}-{sub_end}/{length}"
                        if response.status != 206 or response.headers.get("Content-Range") != expected:
                            raise RuntimeError(f"Server did not honor range: {response.status}, {response.headers}")
                        while cursor <= sub_end:
                            payload = response.read(min(65536, sub_end - cursor + 1))
                            if not payload:
                                raise OSError(f"Incomplete HTTP range at byte {cursor}")
                            written = 0
                            while written < len(payload):
                                written += os.pwrite(fd, payload[written:], cursor + written)
                            cursor += len(payload)
                except (OSError, TimeoutError) as error:
                    # Preserve a validated prefix and request only the missing suffix.
                    failures = 0 if cursor > request_start else failures + 1
                    logger.warning("Range %s resume offset %s, failures=%s: %s", index, cursor, failures, error)
                    if failures >= 6:
                        raise
                    if failures:
                        time.sleep(min(2 ** failures, 15))
        return index, end - start + 1
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(transfer, segment) for segment in segments if segment[0] not in completed]
            for future in as_completed(futures):
                index, size = future.result()
                completed.add(index)
                already += size
                # fsync before recording completion, so resume cannot trust unflushed data.
                os.fsync(fd)
                save_json(state_path, signature | {"completed": sorted(completed)})
                logger.info("Range download %s: %.1f%% (%s/%s bytes)", path.name,
                            already / length * 100, already, length)
    finally:
        os.close(fd)
    partial.replace(path)
    state_path.unlink(missing_ok=True)
    return path


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def download_plan(root=ROOT):
    files=json.loads((root/'configs/data_sources.json').read_text())['files']
    total=sum(f['content_length'] for f in files)
    plan=dict(files=files,total_bytes=total,limit_bytes=LIMIT,
              free_bytes=shutil.disk_usage(root).free,allowed=total<=LIMIT)
    for item in files:
        print(f"PLAN {item['path']}: Content-Length={item['content_length']}",flush=True)
    print(f"PLAN total={total}; free={plan['free_bytes']}; limit={LIMIT}",flush=True)
    save_json(root/'results/download_plan.json',plan)
    if not plan['allowed']:raise RuntimeError('Download budget exceeds 60 GB')
    return plan


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(item, manifest, logger, root=ROOT):
    """Resume partial downloads; validate length, keep hash/timing provenance."""
    path = root / item["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size == item["content_length"]:
        logger.info("Reusing %s (%s bytes)", path, path.stat().st_size)
        return path
    reserve = 2_000_000_000
    partial = path.with_suffix(path.suffix + ".part")
    remaining = item["content_length"] - (partial.stat().st_size if partial.exists() else 0)
    if shutil.disk_usage(root).free < remaining + reserve:
        raise RuntimeError(f"Insufficient drive space for {path}; needs {remaining + reserve} bytes")
    # This HEAD cannot download the object body. A changed size requires a new budget.
    with urlopen(Request(item["url"], method="HEAD"), timeout=90) as response:
        length = response.headers.get("Content-Length")
        logger.info("HEAD %s status=%s Content-Length=%s", item["url"], response.status, length)
        if length is not None and int(length) != item["content_length"]:
            raise RuntimeError(f"Content-Length changed from T1: {item['url']}: {length}")
    started = time.monotonic()
    started_at = datetime.now(timezone.utc)
    if item["content_length"] >= 8 * 1024 * 1024:
        fetch_ranges(item, logger, root)
        manifest[item["path"]] = item | {"sha256": sha256_file(path),
            "elapsed_seconds": time.monotonic() - started,
            "started_at_utc": started_at.isoformat(),
            "wall_elapsed_seconds": (datetime.now(timezone.utc) - started_at).total_seconds(),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "removed_after_filter": False, "method": "16 workers; 2-MiB HTTP requests; 8-MiB checkpoints"}
        save_json(root / "data/raw/download_manifest.json", manifest)
        return path
    for attempt in range(6):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "CGCR-T2-data-pipeline/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            with urlopen(Request(item["url"], headers=headers), timeout=90) as response:
                if response.status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {offset}-"):
                        raise RuntimeError(f"Invalid resume range {content_range}")
                    mode = "ab"
                else:
                    mode, offset = "wb", 0
                last = time.monotonic()
                with partial.open(mode) as stream:
                    while chunk := response.read(1024 * 1024):
                        stream.write(chunk)
                        offset += len(chunk)
                        if offset > item["content_length"]:
                            raise RuntimeError("Response exceeds planned Content-Length")
                        if time.monotonic() - last >= 20:
                            logger.info("Download %s: %.1f%% (%s bytes)", path.name,
                                        offset / item["content_length"] * 100, offset)
                            last = time.monotonic()
            if partial.stat().st_size != item["content_length"]:
                raise OSError(f"Short download: {partial.stat().st_size} bytes")
            partial.replace(path)
            record = item | {"sha256": sha256_file(path),
                             "elapsed_seconds": time.monotonic() - started,
                             "started_at_utc": started_at.isoformat(),
                             "wall_elapsed_seconds": (datetime.now(timezone.utc) - started_at).total_seconds(),
                             "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                             "removed_after_filter": False}
            manifest[item["path"]] = record
            save_json(root / "data/raw/download_manifest.json", manifest)
            logger.info("Downloaded %s: %s bytes, %.1fs", path.name, offset, record["elapsed_seconds"])
            return path
        except (OSError, TimeoutError) as error:
            logger.warning("Download attempt %s for %s: %s", attempt + 1, path.name, error)
            if attempt == 5:
                raise
            time.sleep(min(2 ** attempt, 15))
    raise AssertionError("unreachable")


def study_category(name, rows, plan, manifest, salt, logger, root=ROOT):
    out = root / "data/interim" / name
    out.mkdir(parents=True, exist_ok=True)
    saved = out / "study.json"
    if saved.exists():
        study = json.loads(saved.read_text())
        if study["salt_sha256"] != hashlib.sha256(salt.encode()).hexdigest():
            raise RuntimeError("Review salt changed; remove/rebuild interim outputs explicitly")
        logger.info("Reusing completed %s interim study", name)
        return study
    items = {f["kind"]: f for f in plan["files"] if f["category"] == name}
    owned = {kind: not (root / item["path"]).exists() for kind, item in items.items()}
    raw = {kind: fetch(item, manifest, logger, root) for kind, item in items.items()}
    ids = {row["product_id"] for row in rows}
    started = time.monotonic()
    mapping, conflicts = child_parent_mapping(raw["reviews"], ids, out, logger.info)
    catalog = filter_metadata(raw["metadata"], ids | set(mapping.values()), out, logger.info)
    counts = filter_reviews(raw["reviews"], set(catalog), out, salt, logger.info)
    study, _ = coverage(rows, catalog, mapping, counts, name,
                        sum(item["content_length"] for item in items.values()))
    study.update(review_counts=counts, ambiguous_children=conflicts,
                 salt_sha256=hashlib.sha256(salt.encode()).hexdigest(),
                 processing_seconds=time.monotonic() - started)
    save_json(saved, study)
    for kind, path in raw.items():
        if owned[kind]:
            path.unlink()
            manifest[items[kind]["path"]]["removed_after_filter"] = True
    save_json(root / "data/raw/download_manifest.json", manifest)
    logger.info("Completed coverage %s: %s", name,
                {k: v for k, v in study.items() if k not in ("review_counts", "qualifying_query_ids", "ambiguous_children")})
    return study


def prepare_pilot(name, rows, study, seed, run, root=ROOT):
    interim = root / "data/interim" / name
    catalog = {r["parent_asin"]: r for r in pq.read_table(interim / "metadata.parquet").to_pylist()}
    mapping = {r["asin"]: r["parent_asin"]
               for r in pq.read_table(interim / "asin_parent_map.parquet").to_pylist()}
    pools = coverage(rows, catalog, mapping, study["review_counts"], name, study["download_bytes"])[1]
    ids = study["qualifying_query_ids"]
    splits = split_queries(ids, seed)
    out = root / "data/processed" / name
    (out / "reviews").mkdir(parents=True, exist_ok=True)
    records, per_query = [], []
    for qid in ids:
        candidates = build_candidates(qid, pools[qid], catalog, seed, name)
        records.extend(candidates)
        per_query.append({"query_id": qid, "labels": dict(Counter(r["esci_label"] for r in candidates)),
                          "review_rich_items": sum(study["review_counts"][r["item_id"]] >= 5 for r in candidates),
                          "items": [r["item_id"] for r in candidates]})
    schema = pa.schema([(key, pa.string()) for key in
                       ("query_id", "query", "item_id", "esci_label", "seller_text", "product_id",
                        "product_brand", "source_split")] +
                       [("catalog", pa.struct(list(CATALOG_SCHEMA)[1:])), ("reviews", pa.string()),
                        ("utility_eligible", pa.bool_())])
    table = pa.Table.from_pylist(records, schema=schema)
    pq.write_table(table, out / "candidates.parquet")
    for split, query_ids in splits.items():
        selected = set(query_ids)
        pq.write_table(pa.Table.from_pylist([r for r in records if r["query_id"] in selected], schema=schema),
                       out / f"{split}.parquet")
    save_json(out / "split_ids.json", {"seed": seed, "dev_fraction": 0.3, "query_ids": splits})
    write_jsonl(out / "candidate_labels.jsonl", per_query)
    write_jsonl(run / "per_query.jsonl", per_query)
    selected_items = sorted({r["item_id"] for r in records})
    for parent in selected_items:
        destination = out / "reviews" / f"{parent}.parquet"
        source = interim / "reviews" / f"{parent}.parquet"
        if destination.exists():
            destination.unlink()
        try:
            destination.hardlink_to(source)
        except OSError:
            shutil.copy2(source, destination)
    review_counts = [study["review_counts"][parent] for parent in selected_items]
    denominator = len(selected_items)
    stats = {"category": name, "queries": len(ids), "candidates": len(records),
             "unique_items": denominator, "dev_queries": len(splits["dev"]),
             "test_queries": len(splits["test"]),
             "labels": dict(Counter(r["esci_label"] for r in records)),
             "per_set_label_distributions": dict(Counter(
                 json.dumps(r["labels"], sort_keys=True) for r in per_query)),
             "sets_with_ge3_review_rich_items": sum(r["review_rich_items"] >= 3 for r in per_query),
             "review_rich_items_per_set": dict(Counter(r["review_rich_items"] for r in per_query)),
             "reviews_per_item": {"min": min(review_counts, default=None),
                                   "median": statistics.median(review_counts) if review_counts else None,
                                   "max": max(review_counts, default=None)},
             "nonempty_details_pct": 100 * sum(bool(json.loads(catalog[p]["details"])) for p in selected_items) / denominator if denominator else None,
             "nonempty_features_pct": 100 * sum(bool(catalog[p]["features"]) for p in selected_items) / denominator if denominator else None,
             "with_price_pct": 100 * sum(catalog[p]["price"] not in (None, "", "None", "null") for p in selected_items) / denominator if denominator else None}
    save_json(out / "summary.json", stats)
    return stats


def write_report(studies, selected, stats, run, plan, manifest, root=ROOT):
    run = Path(run)
    if not run.is_absolute():
        run = root / run
    lines = ["# T2 pilot data report", "", f"Generated by `scripts/prepare_data.py`; logged run `{run.relative_to(root)}`.",
             "All counts below are computed from local streaming outputs. No model was run.", "",
             "Counts use distinct parent ASINs per query. Columns form cumulative conditions:",
             "ten joined products; then at least one Exact; then at least three products with five reviews.", "",
             "| Category | Direct ESCI ASIN matches | Additional child matches | Joined parents | ≥10 | ≥10 + Exact | Qualifying | Compressed bytes |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s in studies:
        lines.append("| " + " | ".join(str(s[k]) for k in ("category", "direct_product_matches",
                     "additional_variant_matches", "joined_parent_items", "queries_ge10_joined",
                     "queries_ge10_with_exact", "qualifying_queries", "download_bytes")) + " |")
    lines += ["", f"Automatic pilot: **{selected}**, maximizing qualifying queries; ties use fewer download bytes.",
              "", "Pilot counts and quality (unique selected parents for item percentages):", "", "```json",
              json.dumps(stats, indent=2), "```", "",
              "Per-query label counts and parent IDs are in `candidate_labels.jsonl` and the run's `per_query.jsonl`.",
              "", "Preparation uses seed from config and a floor(30% × query count) dev split; the rest is test.",
              "Original ESCI splits remain `source_split`. Variant duplicates prefer the best ESCI label E/S/C/I,",
              "then lexical ASIN. All Exact parents are kept up to ten; other joined parents are sampled.",
              "The review-rich qualification applies to the joined pool, before sampling.",
              "", "Catalog `details` is lossless JSON text in parquet; price is nullable source text.",
              "Reviews include every variant of each matched parent. User IDs use SHA-256(salt + NUL + ID),",
              "with the fixed salt versioned in config; missing IDs remain null.",
              "Review pointers are relative to the category output directory. Per-item files are hard-linked",
              "from interim data where supported. Ambiguous child mappings are excluded and recorded.",
              "", "Download budget/free-space plan and measured download timings:", "", "```json",
              json.dumps({"plan": plan, "downloads": list(manifest.values())}, indent=2), "```", ""]
    (root / "results/data_report.md").write_text("\n".join(lines))


_download_unchecked=fetch
def fetch(item,manifest,logger,root=ROOT):
    path=_download_unchecked(item,manifest,logger,root)
    actual=sha256_file(path)
    if actual!=item['sha256']:
        raise ValueError('Downloaded file does not match the study SHA-256: '+path.name)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", choices=["auto"] + CATEGORIES, default="auto")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--fetch-esci-only", action="store_true")
    args = parser.parse_args(argv)
    plan = download_plan()
    if args.plan_only:
        return 0
    config = read_yaml(ROOT / "configs/cgcr_default.yaml")
    config["data_preparation"] = {"category": args.category, "download_plan": plan,
                                  "qualification": {"joined": 10, "exact": 1, "review_rich": 3, "reviews": 5}}
    run = start_run("T2_data", config, results_root=ROOT / "results")
    logger = get_logger()
    write_yaml(run / "seeds.yaml", {"seed": config["seed"]})
    save_json(run / "code_manifest.json", {str(p.relative_to(ROOT)): sha256_file(p)
        for p in [Path(__file__).resolve(), ROOT / "configs/cgcr_default.yaml",
                  *sorted((ROOT / "src/cgcr/data").glob("*.py"))]})
    manifest_path = ROOT / "data/raw/download_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    try:
        raw_esci = [fetch(f, manifest, logger) for f in plan["files"] if f["category"] == "ESCI"]
        if args.fetch_esci_only:
            write_jsonl(run / "per_query.jsonl", [])
            save_json(run / "metrics.json", {"stage": "fetch-esci-only", "downloads": manifest})
            return 0
        paths = load_esci(*raw_esci, ROOT / "data/interim/esci")
        rows = read_filtered(*paths)
        logger.info("ESCI retained %s rows, %s unique products, %s queries", len(rows),
                    len({r["product_id"] for r in rows}), len({r["query_id"] for r in rows}))
        names = CATEGORIES if args.category == "auto" else [args.category]
        studies = [study_category(name, rows, plan, manifest, config["data"]["user_id_salt"], logger)
                   for name in names]
        # Preparing one named category must not discard the four-category study.
        coverage_path = ROOT / "data/processed/join_coverage.json"
        previous = json.loads(coverage_path.read_text())["studies"] if coverage_path.exists() else []
        combined = {s["category"]: s for s in previous}
        combined.update({s["category"]: s for s in studies})
        studies = [combined[c] for c in CATEGORIES if c in combined]
        automatic = choose_category(studies)
        selected = automatic if args.category == "auto" else args.category
        save_json(coverage_path, {"studies": studies, "pilot_category": automatic,
                                  "prepared_category": selected})
        stats = prepare_pilot(selected, rows, next(s for s in studies if s["category"] == selected),
                              config["seed"], run)
        metrics = {"pilot": stats, "coverage": studies, "downloads": manifest}
        save_json(run / "metrics.json", metrics)
        write_report(studies, automatic, stats, run, plan, manifest)
        logger.info("Pilot completed: %s", stats)
    except Exception as error:
        logger.exception("T2 pipeline stopped")
        save_json(run / "failure.json", {"error": repr(error), "downloads": manifest})
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
