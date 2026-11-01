"""details-only schema normalization, alias deduplication and exclusions."""
import pytest

from cgcr.data.schema_builder import normalize_key, KeyPolicy, build_schema


@pytest.mark.parametrize('key,expected', [
    ('  ITEM  Weight (lbs) ', 'item weight'), ('Weight [kg]', 'weight'),
    ('Weight lbs', 'weight'), ('Age Range (Description)', 'age range (description)'),
    ('\u200eBatteries Required?', 'batteries required'), (' Color : ', 'color'),
])
def test_key_normalization(key, expected):
    assert normalize_key(key) == expected


def test_details_only_synonyms_item_denominator_and_threshold():
    items = {str(i): {'details': {}, 'features': ['invented feature attribute'], 'title': 'invented title'} for i in range(100)}
    items['0']['details'] = {'Weight': '2 pounds', 'Item Weight': '2 pounds', 'ASIN': 'id', 'Rare': 'v'}
    items['1']['details'] = {'Item Weight (lb)': '3 pounds', 'Assembly Required': 'Yes'}
    items['2']['details'] = {'Assembly Required': 'No'}
    schema = build_schema(items, {'synonyms': {'item weight': ['Weight']}, 'exclude': {'ASIN': 'identifier'}}, 'test')
    attrs = {a['canonical_name']: a for a in schema['attributes']}
    assert set(attrs) == {'item weight', 'assembly required'}
    weight = attrs['item weight']
    assert weight['item_count'] == 2 and weight['coverage_pct'] == 2
    assert weight['value_type'] == 'numeric+unit' and weight['variant_sensitive'] is True
    assert weight['top_10_example_values'][0]['item_count'] == 1
    assert attrs['assembly required']['value_type'] == 'boolean'
    assert schema['excluded_observed_keys'] == {'ASIN': 1}
    assert schema['below_threshold'] == [{'canonical_name': 'rare', 'item_count': 1}]
    with pytest.raises(ValueError, match='Conflicting synonym'):
        KeyPolicy({'synonyms': {'a': ['same'], 'b': ['same']}})


def test_configured_types_ambiguous_observed_units_and_counts():
    from cgcr.data.schema_builder import is_variant_sensitive
    schema = build_schema({'p': {'details': {'Color': 'Black', 'Capacity': '12 Ounces'}}},
                          {'value_types': {'color': 'categorical'}}, 'test')
    attrs = {a['canonical_name']: a for a in schema['attributes']}
    assert attrs['color']['value_type'] == 'categorical'
    assert attrs['color']['value_type_source'] == 'configured'
    assert attrs['capacity']['observed_units'] == []
    assert attrs['capacity']['numeric_parse_pct'] == 0
    assert is_variant_sensitive('number of shelves')
