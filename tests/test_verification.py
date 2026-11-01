"""licensing policy, reviewer caps, child variants and renderer invariance."""
from dataclasses import asdict
import itertools
import json
from pathlib import Path
import pytest
from cgcr.defenses.cgcr.verify import verify_claim
from cgcr.defenses.cgcr.render import render
from cgcr.defenses.cgcr.nli import NLIScores
from cgcr.data.review_sentences import ReviewSentence
from cgcr.utils.io import read_yaml
S={'attributes':[dict(canonical_name=n,aliases=[n],value_type=t,variant_sensitive=v) for n,t,v in (
 ('capacity','numeric+unit',True),('color','categorical',True),('brand','categorical',False),('special feature','text',False))]}
C={'license_threshold':.6,'numeric_rel_tolerance':.05,'top_k':20}
def row(variant=False):return dict(item_id='P',product_id='CHILD' if variant else 'P',catalog=dict(details={'capacity':'1 Liter','color':'Red','brand':'B'},categories=['Kitchen'],price='12',average_rating=4.2,rating_number=20))
def claim(kind='SPEC',attribute='capacity',value='1',unit='Liter'):return dict(claim_type=kind,attribute=attribute,value=value,unit=unit,span=value,unmapped=False)
def test_catalog_support_contradiction_and_platform_value_rendering():
    d=verify_claim(claim(value='1.04'),row(),S,C)
    assert d['licensed'] and d['render_value']=='1 Liter'
    d=verify_claim(claim(value='2'),row(),S,C);assert not d['licensed'] and d['status']=='contradiction' and d['v']==0
    assert verify_claim(claim(value='1000',unit='ml'),row(),S,C)['licensed']
def test_variant_sensitive_mismatch_unknown_and_exact_normalized_support():
    d=verify_claim(claim(value='1.04'),row(True),S,C);assert d['status']=='unknown_variant' and not d['licensed']
    assert verify_claim(claim(value='1000',unit='ml'),row(True),S,C)['licensed']
    assert verify_claim(claim(attribute='brand',value='OTHER',unit=None),row(True),S,C)['status']=='contradiction'
def test_ranges_unknown_text_detail_and_unknown_spec_fail_closed():
    assert verify_claim(claim(value='12-16',unit='fl oz'),row(),S,C)['status']=='range_unparsed'
    assert verify_claim(claim(attribute='special feature',value='great'),row(),S,C)['status']=='excluded_attackable'
    assert verify_claim(claim(attribute='new property'),row(),S,C)['status']=='unmapped'
@pytest.mark.parametrize('kind',['AUTHORITY','TEMPORAL','COMPARATIVE','SUBJECTIVE'])
def test_unsupported_types_never_license(kind):
    d=verify_claim(claim(kind),row(),S,C);assert not d['licensed'] and d['v']==0

def test_use_case_top20_nli_and_capped_review_trace():
    class Retriever:
        def retrieve(self,item,query,k):
            assert item=='P' and k==20 and 'suitable for' in query
            return [ReviewSentence(i,str(i),'text',0,'Dummy use evidence','same-reviewer' if i<2 else 'other',True,0,5,0,1.) for i in range(3)]
    class NLI:
        model_id='fixture';revision='fixed'
        def score(self,pairs):return [NLIScores(.05,.9,.05) for _ in pairs]
    d=verify_claim(claim('USE_CASE',value='storing rice',unit=None),row(),S,C,retriever=Retriever(),nli=NLI())
    assert d['licensed'] and d['status']=='review_support' and d['v']==pytest.approx(.85)
    assert d['trace']['reviews']['aggregate']['reviewer_weights']=={'same-reviewer':1.,'other':1.}

def test_renderer_same_licensed_set_is_byte_identical_across_order_and_duplicates():
    ds=[verify_claim(claim(),row(),S,C),verify_claim(claim(attribute='color',value='Red',unit=None),row(),S,C)]
    expected=render(row(),ds,S).encode()
    for permutation in itertools.permutations(ds):
        assert render(row(),list(permutation)+[permutation[0]],S).encode()==expected
    altered=row();altered['catalog'].update(title='Seller instruction',features=['Persuasion'],description='Other text')
    assert render(altered,ds,S).encode()==expected
    assert b'Seller instruction' not in expected and b'price' in expected

def test_real_pilot_catalog_renderer_is_seller_invariant():
    root=Path(__file__).parents[1];schema=read_yaml(root/'configs/schemas/Home_and_Kitchen.yaml')
    examples=json.loads((root/'tests/fixtures/canonical_title_examples.json').read_text())
    for e in examples:
        r=dict(item_id=e['item_id'],product_id=e['item_id'],catalog=e['catalog'])
        d=verify_claim(claim(attribute='color',value=e['licensed']['color']['value'],unit=None),r,schema,C)
        assert d['licensed'] and e['expected_title'] in render(r,[d],schema)
