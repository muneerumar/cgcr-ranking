"""offline fixtures exercise filters, variant joins, and pilot invariants."""

import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cgcr.data.esci import load_esci, read_filtered, seller_text
from cgcr.data.amazon_reviews_2023 import (
    child_parent_mapping, filter_metadata, filter_reviews, iter_filtered, hash_user_id,
)
from cgcr.data.candidate_sets import (
    build_candidates, choose_category, coverage, joined_pools, split_queries,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_esci_us_small_filter_and_seller_join(tmp_path):
    examples = [dict(query_id=1, query="phone", product_id="P00", product_locale="us",
                     esci_label="E", split="train", small_version=1),
                dict(query_id=2, query="skip", product_id="P01", product_locale="us",
                     esci_label="S", split="test", small_version=0),
                dict(query_id=3, query="skip", product_id="P00", product_locale="es",
                     esci_label="I", split="train", small_version=1)]
    products = [dict(product_id="P00", product_locale=locale, product_title=title,
                     product_description="description", product_bullet_point="bullets", product_brand="brand")
                for locale, title in [("us", "US title"), ("es", "Spanish title")]]
    ex, pr = tmp_path / "examples.parquet", tmp_path / "products.parquet"
    pq.write_table(pa.Table.from_pylist(examples), ex)
    pq.write_table(pa.Table.from_pylist(products), pr)
    rows = read_filtered(*load_esci(ex, pr, tmp_path / "filtered", batch_size=1))
    assert len(rows) == 1
    assert rows[0]["query_id"] == "1" and rows[0]["split"] == "train"
    assert seller_text(rows[0]) == "US title\nbullets\ndescription"
    assert rows[0]["product_brand"] == "brand"


def test_streaming_filter_and_parent_mapping(tmp_path):
    reviews = FIXTURES / "reviews_test.jsonl.gz"
    assert {r["asin"] for r in iter_filtered(reviews, {"C00"}, "asin")} == {"C00"}
    mapping, conflicts = child_parent_mapping(reviews, {"C00", "C01", "AMBIG"}, tmp_path)
    assert mapping == {"C00": "P00", "C01": "P01"}
    assert conflicts == ["AMBIG"]
    catalog = filter_metadata(FIXTURES / "meta_test.jsonl.gz", {"P00", "P01"}, tmp_path)
    assert set(catalog) == {"P00", "P01"}
    assert json.loads(catalog["P00"]["details"]) == {"material": "metal"}
    counts = filter_reviews(reviews, {"P00", "P01", "EMPTY"}, tmp_path, "fixed-salt")
    assert counts == {"P00": 6, "P01": 1, "EMPTY": 0}
    retained = pq.read_table(tmp_path / "reviews/P00.parquet").to_pylist()
    # Includes a different child's review that appears before C00 is discovered.
    assert retained[0]["text"] == "Earlier variant review"
    assert retained[0]["user_id"] == hash_user_id("raw-user", "fixed-salt")
    assert retained[0]["user_id"] != "raw-user"
    assert hash_user_id(None, "fixed-salt") is None
    assert hash_user_id("raw-user", "different") != retained[0]["user_id"]
    assert pq.read_table(tmp_path / "reviews/EMPTY.parquet").num_rows == 0
    assert not (tmp_path / "reviews_spool.sqlite").exists()


def pool_fixture():
    rows, catalog = [], {}
    for index in range(13):
        parent = f"P{index:02}"
        catalog[parent] = dict(title=parent, features=["feature"], details="{}", price="1.00",
                               average_rating=4.0, rating_number=7, categories=["test"], store="store")
        rows.append(dict(query_id="q", query="test", product_id=parent, item_id=parent,
                         esci_label="E" if index < 2 else "S", split="train",
                         product_title=parent, product_bullet_point="", product_description=""))
    return rows, catalog


def test_candidate_invariants_exact_first_deterministic():
    pool, catalog = pool_fixture()
    result = build_candidates("q", pool, catalog, 13, "test")
    assert len(result) == 10 and len({r["item_id"] for r in result}) == 10
    assert [r["item_id"] for r in result[:2]] == ["P00", "P01"]
    assert all(r["esci_label"] != "E" for r in result[2:])
    assert result == build_candidates("q", list(reversed(pool)), catalog, 13, "test")
    assert all(r["reviews"] == f"reviews/{r['item_id']}.parquet" for r in result)
    with pytest.raises(ValueError, match="unique"):
        build_candidates("q", pool + pool[:1], catalog)
    with pytest.raises(ValueError, match="ten"):
        build_candidates("q", pool[:9], catalog)
    for row in pool:
        row["esci_label"] = "E"
    assert len(build_candidates("q", pool, catalog)) == 10


def test_parent_dedup_coverage_and_pilot_choice():
    rows, catalog = pool_fixture()
    rows.append(rows[0] | {"product_id": "CHILD", "esci_label": "I"})
    # Directs plus one additional child, but the same number of distinct parents.
    result, pools = coverage(rows, catalog, {"CHILD": "P00"},
                             {"P00": 5, "P01": 5, "P02": 5}, "A", 100)
    assert result["direct_product_matches"] == 13
    assert result["additional_variant_matches"] == 1
    assert result["qualifying_queries"] == 1
    assert len(pools["q"]) == 13
    assert pools["q"][0]["esci_label"] == "E"
    other = result | {"category": "B", "download_bytes": 50}
    assert choose_category([result, other]) == "B"
    assert choose_category([result | {"qualifying_queries": 2}, other]) == "A"
    # Many child variants must not make a query qualify as ten parents.
    duplicates = [rows[0] | {"product_id": f"CH{i}"} for i in range(12)]
    result, _ = coverage(duplicates, catalog, {r["product_id"]: "P00" for r in duplicates},
                         {"P00": 99}, "A", 100)
    assert result["queries_ge10_joined"] == 0


def test_deterministic_split():
    ids = [str(i) for i in range(20)]
    splits = split_queries(ids, 13)
    assert splits == split_queries(reversed(ids), 13)
    assert len(splits["dev"]) == 6 and len(splits["test"]) == 14
    assert set(splits["dev"]).isdisjoint(splits["test"])
    assert set(splits["dev"] + splits["test"]) == set(ids)




def test_ranged_download_validates_and_reassembles(tmp_path, monkeypatch):
    import io
    import logging
    script = Path(__file__).parents[1] / "scripts/prepare_data.py"
    spec = importlib.util.spec_from_file_location("prepare_ranges_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload = b"a tiny range test with several chunks"
    requests = []

    def opener(request, **kwargs):
        start, end = map(int, request.headers["Range"].removeprefix("bytes=").split("-"))
        requests.append((start, end))
        result = io.BytesIO(payload[start:end + 1])
        result.status = 206
        result.headers = {"Content-Range": f"bytes {start}-{end}/{len(payload)}"}
        return result

    monkeypatch.setattr(module, "urlopen", opener)
    item = dict(path="tiny.bin", url="https://example.test/tiny", content_length=len(payload))
    path = module.fetch_ranges(item, logging.getLogger("test"), tmp_path, workers=2, chunk_size=8)
    assert path.read_bytes() == payload
    assert len(requests) == 5
    assert not list(tmp_path.glob("*.part")) and not list(tmp_path.glob("*.json"))


def test_prepared_pilot_outputs_and_generated_counts(tmp_path):
    script = Path(__file__).parents[1] / "scripts/prepare_data.py"
    spec = importlib.util.spec_from_file_location("prepare_pilot_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows, catalog = pool_fixture()
    interim = tmp_path / "data/interim/Fixture"
    interim.mkdir(parents=True)
    from cgcr.data.amazon_reviews_2023 import CATALOG_SCHEMA, REVIEW_SCHEMA
    pq.write_table(pa.Table.from_pylist([r | {"parent_asin": parent} for parent, r in catalog.items()],
                                       schema=CATALOG_SCHEMA), interim / "metadata.parquet")
    pq.write_table(pa.table({"asin": pa.array([], pa.string()), "parent_asin": pa.array([], pa.string())}),
                   interim / "asin_parent_map.parquet")
    (interim / "reviews").mkdir()
    for parent in catalog:
        review = dict(rating=4.0, title="fixture", text="fixture review", timestamp=1,
                      verified_purchase=True, helpful_vote=0, user_id=hash_user_id("fixture", "salt"))
        pq.write_table(pa.Table.from_pylist([review] * 5, schema=REVIEW_SCHEMA),
                       interim / "reviews" / f"{parent}.parquet")
    study = dict(qualifying_query_ids=["q"], review_counts={p: 5 for p in catalog}, download_bytes=100)
    run = tmp_path / "run"
    run.mkdir()
    stats = module.prepare_pilot("Fixture", rows, study, 13, run, tmp_path)
    out = tmp_path / "data/processed/Fixture"
    result = pq.read_table(out / "candidates.parquet").to_pylist()
    assert len(result) == 10 and stats["queries"] == 1
    assert stats["labels"] == {"E": 2, "S": 8}
    assert stats["sets_with_ge3_review_rich_items"] == 1
    assert stats["review_rich_items_per_set"] == {10: 1}
    assert stats["nonempty_features_pct"] == 100 and stats["nonempty_details_pct"] == 0
    assert json.loads((out / "split_ids.json").read_text())["query_ids"] == {"dev": [], "test": ["q"]}
    assert all((out / r["reviews"]).is_file() for r in result)
    assert all(pq.read_metadata(out / r["reviews"]).num_rows == 5 for r in result)
    assert pq.read_table(out / "dev.parquet").num_rows == 0


def test_download_resumes_a_truncated_response_without_repeating_prefix(tmp_path, monkeypatch):
    import io
    import logging
    script = Path(__file__).parents[1] / "scripts/prepare_data.py"
    spec = importlib.util.spec_from_file_location("prepare_partial_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload = bytes(range(256)) * 8192
    requested = []

    def opener(request, **kwargs):
        start, end = map(int, request.headers["Range"].removeprefix("bytes=").split("-"))
        requested.append(start)
        body = payload[start:end + 1]
        if len(requested) == 1:
            body = body[:len(body) // 2]
        response = io.BytesIO(body)
        response.status = 206
        response.headers = {"Content-Range": f"bytes {start}-{end}/{len(payload)}"}
        return response

    monkeypatch.setattr(module, "urlopen", opener)
    item = dict(path="partial.bin", url="https://example.test/partial", content_length=len(payload))
    path = module.fetch_ranges(item, logging.getLogger("test"), tmp_path, workers=1)
    assert requested == [0, len(payload) // 2]
    assert path.read_bytes() == payload


