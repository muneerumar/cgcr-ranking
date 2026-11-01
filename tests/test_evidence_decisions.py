"""typed evidence, variant mismatch routing, ranges, and real neutral titles."""
import json
from pathlib import Path

import pytest
from cgcr.data.evidence import canonical_title, classify_detail, detail_match_status, evidence_details, catalog_evidence
from cgcr.data.units import unit_parse_status, parse_volume
from cgcr.utils.io import read_yaml

SCHEMA=read_yaml(Path(__file__).parents[1]/'configs/schemas/Home_and_Kitchen.yaml')
EXAMPLES=json.loads((Path(__file__).parent/'fixtures/canonical_title_examples.json').read_text())


def test_only_typed_details_are_evidence():
    for name in ('special feature','recommended uses for product','specific uses for product','warranty description','included components'):
        assert classify_detail(name,SCHEMA)=='ATTACKABLE'
    assert classify_detail('color',SCHEMA)=='EVIDENCE'
    assert classify_detail('unknown',SCHEMA)=='EXCLUDED'
    catalog={'details':json.dumps({'Brand':'Brand','Color':'Red','Special Feature':'prose','Unknown':'value'})}
    assert evidence_details(catalog,SCHEMA)=={'brand':['Brand'],'color':['Red']}
    assert json.loads(catalog_evidence(catalog)['details'])=={}  # schema omission fails closed


def test_variant_and_range_policy():
    assert detail_match_status('color','Red','Red',SCHEMA,joined_via_variant=True)=='support'
    assert detail_match_status('color','Blue','Red',SCHEMA,joined_via_variant=True)=='unknown_variant'
    assert detail_match_status('color','Blue','Red',SCHEMA)=='contradiction'
    assert detail_match_status('brand','B','A',SCHEMA,joined_via_variant=True)=='contradiction'
    assert detail_match_status('color','Blue','Red',SCHEMA,joined_via_variant=True,review_supported=True)=='review_support'
    assert detail_match_status('capacity','1 L','1000 ml',SCHEMA,joined_via_variant=True)=='support'
    assert detail_match_status('capacity','12-16 fl oz','12-16 fl oz',SCHEMA)=='range_unparsed'
    assert unit_parse_status('12–16 oz','item weight')=='range_unparsed'
    assert unit_parse_status('-40 C','temperature')=='parsed'
    assert parse_volume('1 cup').values==pytest.approx((236.5882365,))


@pytest.mark.parametrize('example',EXAMPLES,ids=lambda r:r['item_id'])
def test_real_titles_and_seller_invariance(example):
    assert example['source']=='T3 fixed dev pilot catalog'
    catalog=example['catalog']
    assert canonical_title(catalog,SCHEMA,example['licensed'])==example['expected_title']
    altered=catalog|{'title':'Changed seller title','features':['Changed prose'],'description':'changed'}
    assert canonical_title(altered,SCHEMA,example['licensed'])==example['expected_title']
    assert canonical_title(catalog,SCHEMA)==example['expected_title'].split(' | color:')[0]
    assert canonical_title(catalog,SCHEMA,{'color':{'value':'different','status':'unknown_variant'}})==canonical_title(catalog,SCHEMA)


def test_units_only_on_numeric_attributes():
    assert all(a['observed_units']==[] for a in SCHEMA['attributes'] if a['value_type']!='numeric+unit')
