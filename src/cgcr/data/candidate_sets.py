"""parent-deduplicated ESCI pools, coverage, candidates, and query splits."""

from collections import Counter, defaultdict
import hashlib
import random

from cgcr.data.esci import seller_text
from cgcr.data.amazon_reviews_2023 import CATALOG_FIELDS

LABEL_ORDER = {"E": 0, "S": 1, "C": 2, "I": 3}


def joined_pools(rows, catalog, mapping):
    """Prefer direct parents; collapse variant duplicates by best ESCI label.

    Ties use lexicographic source ASIN, independent of parquet input ordering.
    """
    pools = defaultdict(dict)
    for row in rows:
        asin = row["product_id"]
        parent = asin if asin in catalog else mapping.get(asin)
        if parent not in catalog:
            continue
        query = str(row["query_id"])
        candidate = row | {"item_id": parent}
        previous = pools[query].get(parent)
        rank = lambda r: (LABEL_ORDER[r["esci_label"]], r["product_id"])
        if previous is None or rank(candidate) < rank(previous):
            pools[query][parent] = candidate
    return {qid: sorted(items.values(), key=lambda r: (LABEL_ORDER[r["esci_label"]], r["item_id"]))
            for qid, items in pools.items()}


def coverage(rows, catalog, mapping, review_counts, category, download_bytes):
    """Measure qualification on distinct joined parents, before candidate sampling."""
    ids = {row["product_id"] for row in rows}
    direct = ids & catalog.keys()
    extra = {asin for asin in ids - direct if mapping.get(asin) in catalog}
    pools = joined_pools(rows, catalog, mapping)
    at_least_ten = [q for q, pool in pools.items() if len(pool) >= 10]
    with_exact = [q for q in at_least_ten if any(r["esci_label"] == "E" for r in pools[q])]
    qualified = [q for q in with_exact
                 if sum(review_counts.get(r["item_id"], 0) >= 5 for r in pools[q]) >= 3]
    result = {"category": category, "download_bytes": download_bytes,
              "direct_product_matches": len(direct), "additional_variant_matches": len(extra),
              "joined_parent_items": len(catalog), "queries_ge10_joined": len(at_least_ten),
              "queries_ge10_with_exact": len(with_exact), "qualifying_queries": len(qualified),
              "qualifying_query_ids": sorted(qualified)}
    return result, pools


def choose_category(studies):
    """Maximize qualifying queries; ties favor smaller compressed download."""
    if not studies:
        raise ValueError("No completed categories to compare")
    return min(studies, key=lambda s: (-s["qualifying_queries"], s["download_bytes"], s["category"]))["category"]


def build_candidates(query_id, pool, catalog, seed=13, category=""):
    """Keep Exact candidates first (up to ten), then sample other joined parents."""
    if len({r["item_id"] for r in pool}) != len(pool):
        raise ValueError("Pool must contain unique parent items")
    if len(pool) < 10 or not any(r["esci_label"] == "E" for r in pool):
        raise ValueError("A candidate set needs ten joined items and an Exact")
    digest = hashlib.sha256(f"{seed}\0{category}\0{query_id}".encode()).digest()
    rng = random.Random(int.from_bytes(digest, "big"))
    ordered = sorted(pool, key=lambda r: r["item_id"])
    exact = [r for r in ordered if r["esci_label"] == "E"]
    exact = sorted(rng.sample(exact, 10), key=lambda r: r["item_id"]) if len(exact) > 10 else exact
    remaining = [r for r in ordered if r["esci_label"] != "E"]
    selected = exact + rng.sample(remaining, 10 - len(exact))
    return [{"query_id": str(query_id), "query": row["query"], "item_id": row["item_id"],
             "esci_label": row["esci_label"], "seller_text": seller_text(row),
             "product_id": row["product_id"], "product_brand": row.get("product_brand", ""),
             "source_split": row["split"],
             "catalog": {key: catalog[row["item_id"]][key] for key in CATALOG_FIELDS},
             "reviews": f"reviews/{row['item_id']}.parquet",
             "utility_eligible": any(r["esci_label"] != "E" for r in selected)}
            for row in selected]


def split_queries(query_ids, seed=13):
    """Shuffle unique sorted IDs; floor 30% dev, remaining 70% test."""
    ids = sorted(set(map(str, query_ids)))
    random.Random(seed).shuffle(ids)
    count = int(len(ids) * 0.3)
    return {"dev": sorted(ids[:count]), "test": sorted(ids[count:])}


def utility_eligibility(rows):
    """mixed-label query sets are eligible for ranking-utility evaluation."""
    eligible = {}
    for row in rows:
        query = str(row['query_id'])
        eligible[query] = eligible.get(query, False) or row['esci_label'] != 'E'
    return eligible


def annotate_utility(rows):
    """repeat a query-level eligibility boolean on every candidate row."""
    eligible = utility_eligibility(rows)
    return [row | {'utility_eligible': eligible[str(row['query_id'])]} for row in rows]


def select_pilot(dev_ids, eligibility, seed=13, size=300, min_utility=200):
    """proportional two-stratum dev sample with an eligibility lower bound.

    Input sorting, separate seeded strata, and a sorted output make membership
    invariant to input row/order. Insufficient strata raise before writing data.
    """
    dev = sorted(set(map(str, dev_ids)))
    if size <= 0 or not 0 <= min_utility <= size or len(dev) < size:
        raise ValueError('Invalid pilot size or insufficient dev queries')
    if any(q not in eligibility for q in dev):
        raise ValueError('Missing dev eligibility')
    eligible = [q for q in dev if eligibility[q]]
    other = [q for q in dev if not eligibility[q]]
    if len(eligible) < min_utility:
        raise ValueError('Not enough utility-eligible dev queries')
    target = max(min_utility, size * len(eligible) // len(dev), size - len(other))
    target = min(target, size, len(eligible))
    rng = random.Random(seed)
    selected = rng.sample(eligible, target) + rng.sample(other, size - target)
    return sorted(selected)
