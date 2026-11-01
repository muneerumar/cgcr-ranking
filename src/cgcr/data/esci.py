"""bounded-batch ESCI US/small loaders and seller-text joins."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

EXAMPLE_FIELDS = ["query_id", "query", "product_id", "esci_label", "split"]
PRODUCT_FIELDS = ["product_id", "product_title", "product_description",
                  "product_bullet_point", "product_brand"]


def text(value):
    """Normalize nullable ESCI text without manufacturing content."""
    return "" if value is None else str(value)


def load_esci(examples_path, products_path, output_dir, batch_size=4096):
    """Filter raw parquet in batches; join only the retained US seller records.

    Returns filtered paths. The complete raw product table is never materialized.
    Intermediate examples retain the source split independently of 's query split.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    example_out, product_out = output_dir / "examples.parquet", output_dir / "products.parquet"
    ids = set()
    example_schema = pa.schema([(key, pa.string()) for key in EXAMPLE_FIELDS])
    with pq.ParquetWriter(example_out, example_schema) as writer:
        for batch in pq.ParquetFile(examples_path).iter_batches(batch_size=batch_size):
            kept = []
            for row in batch.to_pylist():
                if row["product_locale"] == "us" and row["small_version"] == 1:
                    result = {key: text(row[key]) for key in EXAMPLE_FIELDS}
                    kept.append(result)
                    ids.add(result["product_id"])
            if kept:
                writer.write_table(pa.Table.from_pylist(kept, schema=example_schema))
    product_schema = pa.schema([(key, pa.string()) for key in PRODUCT_FIELDS])
    with pq.ParquetWriter(product_out, product_schema) as writer:
        for batch in pq.ParquetFile(products_path).iter_batches(batch_size=batch_size):
            kept = [{key: text(row.get(key)) for key in PRODUCT_FIELDS}
                    for row in batch.to_pylist()
                    if row["product_locale"] == "us" and row["product_id"] in ids]
            if kept:
                writer.write_table(pa.Table.from_pylist(kept, schema=product_schema))
    return example_out, product_out


def read_filtered(examples_path, products_path):
    """Load the bounded US/small subset, joining on the already-filtered locale."""
    products = {r["product_id"]: r for batch in pq.ParquetFile(products_path).iter_batches()
                for r in batch.to_pylist()}
    rows = []
    for batch in pq.ParquetFile(examples_path).iter_batches():
        for row in batch.to_pylist():
            if row["product_id"] in products:
                rows.append(row | products[row["product_id"]])
    return rows


def seller_text(row):
    """Join only the requested ESCI title, bullets, and description."""
    return "\n".join(text(row.get(key)).strip() for key in
                     ("product_title", "product_bullet_point", "product_description")
                     if text(row.get(key)).strip())
