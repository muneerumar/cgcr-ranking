"""source classification, structured whitelist, overlap and variant provenance."""
import json

import pytest

from cgcr.data.evidence import classify_field, catalog_evidence, evidence_record, overlap_audit
from cgcr.data.schema_builder import KeyPolicy


@pytest.mark.parametrize('source,field,expected', [
    ('esci', 'product_title', 'ATTACKABLE'), ('esci', 'product_bullet_point', 'ATTACKABLE'),
    ('esci', 'product_description', 'ATTACKABLE'), ('esci', 'seller_text', 'ATTACKABLE'),
    ('catalog', 'title', 'ATTACKABLE'), ('catalog', 'features', 'ATTACKABLE'),
    ('catalog', 'description', 'ATTACKABLE'), ('catalog', 'details', 'EVIDENCE'),
    ('catalog', 'price', 'EVIDENCE'), ('catalog', 'average_rating', 'EVIDENCE'),
    ('catalog', 'rating_number', 'EVIDENCE'), ('catalog', 'categories', 'EVIDENCE'),
    ('reviews', 'title', 'EVIDENCE'), ('reviews', 'text', 'EVIDENCE'),
    ('catalog', 'store', 'EXCLUDED'), ('esci', 'product_brand', 'EXCLUDED'),
    ('new_source', 'details', 'EXCLUDED'), ('catalog', 'unknown', 'EXCLUDED'),
])
def test_classification(source, field, expected):
    assert classify_field(source, field) == expected


def test_evidence_whitelist_and_variant_flags():
    policy = KeyPolicy({'synonyms': {'item weight': ['Weight']}, 'exclude': {'ASIN': 'identifier'}})
    catalog = {'title': 'seller title', 'features': ['seller feature'], 'description': 'seller prose',
               'details': json.dumps({'Weight': '2 pounds', 'Color': 'Red', 'Size': 'Large', 'Capacity': '2 L',
                                     'Unit Count': '2 Count', 'Product Dimensions': '1 x 2 x 3 inches',
                                     'ASIN': 'not-attribute', 'Brand': 'brand'}),
               'price': '12', 'average_rating': 4, 'rating_number': 5, 'categories': ['kitchen'], 'store': 'seller'}
    row = {'query_id': 'q', 'item_id': 'P', 'product_id': 'C', 'catalog': catalog, 'reviews': 'reviews/P.parquet'}
    evidence = evidence_record(row, policy)
    assert evidence['joined_via_variant'] is True
    assert evidence['variant_sensitive_keys'] == ['capacity', 'color', 'item weight', 'product dimensions', 'size', 'unit count']
    assert set(catalog_evidence(catalog)) == {'details', 'price', 'average_rating', 'rating_number', 'categories'}
    assert evidence_record(row | {'product_id': 'P'}, policy)['joined_via_variant'] is False
    assert not {'features', 'title', 'description', 'store'} & evidence['catalog'].keys()


def test_overlap_item_weighted_exact_case_and_short_values():
    base = {'item_id': 'P', 'catalog': {'features': ['Full feature', 'second'], 'details': '{"Color":"Red", "Count":1}'}}
    rows = [base | {'seller_text': 'Full feature with 1 item'},
            base | {'seller_text': 'second is red'},
            {'item_id': 'Q', 'seller_text': 'full FEATURE', 'catalog': {'features': ['Full feature'], 'details': '{}'}}]
    metrics, records = overlap_audit(rows)
    assert metrics['unique_items'] == 2
    assert metrics['feature_verbatim_overlap']['pct'] == 50
    assert metrics['all_features_verbatim_overlap']['items'] == 1
    assert metrics['details_value_verbatim_overlap']['items'] == 1
    assert metrics['details_value_ge4chars_verbatim_overlap']['items'] == 0
    assert records[1]['feature_verbatim_overlap'] is False
    assert overlap_audit([])[0]['feature_verbatim_overlap']['pct'] is None
    with pytest.raises(ValueError, match='Conflicting parent'):
        overlap_audit(rows + [base | {'catalog': {'features': [], 'details': '{}'}, 'seller_text': ''}])
