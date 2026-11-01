"""parsers checked against provenance-tagged real local catalog strings."""
import json
from pathlib import Path

import pytest

from cgcr.data.units import parse_attribute, parse_dimensions, parse_temperature, parse_volume, parse_weight

REAL = json.loads((Path(__file__).parent / 'fixtures/catalog_unit_examples.json').read_text())


@pytest.mark.parametrize('example', REAL, ids=lambda ex: ex['attribute'])
def test_real_catalog_examples(example):
    assert example['item_id'] and example['source'] == 'T2 Home_and_Kitchen catalog.details'
    result = parse_attribute(example['value'], example['attribute'])
    if example.get('unparsed'):
        assert result is None
    else:
        assert result.values == pytest.approx(example['expected_values'])
        assert result.unit == example['expected_unit']
        if 'expected_axes' in example:
            assert result.axes == tuple(example['expected_axes'])


@pytest.mark.parametrize('value', ['12 Ounces', '500', '2 liters extra prose', '1-2 liters', '-3 L'])
def test_ambiguous_or_invalid_volume_unknown(value):
    assert parse_volume(value) is None


def test_unit_equivalence_axes_and_unknowns():
    assert parse_weight('1 kg').values == parse_weight('1000 grams').values
    assert parse_volume('1 L').values == parse_volume('1000 ml').values
    assert parse_temperature('32 °F').values == pytest.approx((0,))
    assert parse_temperature('273.15 K').values == pytest.approx((0,))
    assert parse_temperature('-500 C') is None
    assert parse_dimensions('1 x 2 x 3 inches').axes == ('', '', '')
    assert parse_dimensions('1"W x 2"W') is None
    assert parse_dimensions('1 x 2 x 3') is None
    assert parse_dimensions('1 x 2 x 3 x 4 inches') is None
    assert parse_weight('2.29 pounds extra') is None


@pytest.mark.parametrize('value,count', [('16 Pieces',16),('Set of 4',4),('2-Pack',2),('3 items',3),('Pack of 5',5)])
def test_explicit_count_conventions(value,count):
    assert parse_attribute(value,'number of items').values==(count,)
    assert parse_attribute(value,'number of pieces').unit=='count'


@pytest.mark.parametrize('value',['-2 pieces','2.5 count','2-4 pieces','Set of 4 extra','2 Count (Pack of 3)'])
def test_ambiguous_or_invalid_counts_remain_unknown(value):
    assert parse_attribute(value,'unit count') is None
