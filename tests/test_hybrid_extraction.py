"""extractor v2: deterministic SPEC recall and source-grounded short line parsing."""
from copy import deepcopy
import pytest
from cgcr.defenses.cgcr.hybrid_extract import deterministic_specs,parse_non_spec,HybridExtractor
from cgcr.defenses.cgcr.catalog import check_catalog
from cgcr.rankers.llm_listwise import Generation


@pytest.fixture
def schema():
    return {'attributes':[dict(canonical_name=n,aliases=a,value_type=t,variant_sensitive=True) for n,a,t in [
        ('capacity',['Volume'],'numeric+unit'),('item weight',['Weight'],'numeric+unit'),
        ('product dimensions',['Dimensions'],'numeric+unit'),('material',['Material'],'categorical'),
        ('special feature',['Feature'],'text')]]}


def test_numeric_conversion_and_nonmatching_inflated_value_both_survive(schema):
    catalog={'details':{'Volume':'2 Liters','Weight':'2 pounds'}}
    claims=deterministic_specs('Capacity 2000 ml; item weight: 9 pounds.',catalog,schema)
    by_name={c['attribute']:c for c in claims}
    assert check_catalog('capacity',by_name['capacity']['value'],catalog,schema).status=='support'
    assert by_name['item weight']['value']=='9 pounds'
    assert check_catalog('item weight',by_name['item weight']['value'],catalog,schema).status=='contradiction'


def test_attacker_unknown_measurements_are_not_dropped(schema):
    claims=deterministic_specs('Power is 250 watts; capacity 12-16 oz.',{'details':{}},schema)
    assert len(claims)==2
    assert all(c['unmapped'] for c in claims)
    assert any('250 watts'==c['span'] for c in claims)
    assert any('12-16 oz'==c['span'] for c in claims)


def test_categorical_match_requires_source_and_excludes_free_text_details(schema):
    catalog={'details':{'Material':'stainless steel','Feature':'beautiful'}}
    claims=deterministic_specs('Beautiful stainless steel body.',catalog,schema)
    assert [c['attribute'] for c in claims]==['material']
    assert deterministic_specs('An unrelated product.',catalog,schema)==[]


def test_negated_property_does_not_become_positive_support(schema):
    catalog={'details':{'Material':'cotton','Volume':'2 liters'}}
    claims=deterministic_specs('Not cotton. Capacity is up to 2 liters.',catalog,schema)
    assert all(check_catalog(c['attribute'],c['value'],catalog,schema).status!='support' for c in claims)


def test_dimension_expression_and_hyphenated_unit(schema):
    catalog={'details':{'Dimensions':'6.5"D x 13.5"W x 6.5"H','Volume':'2 cups'}}
    text='Dimensions: 6.5"D x 13.5"W x 6.5"H. A 2-cup vessel.'
    claims=deterministic_specs(text,catalog,schema)
    assert len(claims)==2
    assert {c['attribute'] for c in claims}=={'product dimensions','capacity'}
    assert all(c['span'] in text for c in claims)


def test_line_parser_rejects_spec_hallucinations_and_token_cap():
    text='For kneading dough. An elegant finish.'
    claims,errors=parse_non_spec('USE_CASE | For kneading dough\nSUBJECTIVE | An elegant finish',text,30)
    assert len(claims)==2 and not errors
    claims,errors=parse_non_spec('SPEC | dough\nUSE_CASE | made-up phrase',text,20)
    assert not claims and len(errors)==2
    assert parse_non_spec('NONE',text,256)[1]==['token_limit']
    assert parse_non_spec('',text)[1]==['empty_output']


def test_hybrid_one_short_call_and_private_cache_identity(schema,tmp_path):
    class Backend:
        calls=0
        def generate(self,messages,**options):
            self.calls+=1
            assert options==dict(seed=13,max_tokens=256,temperature=0.)
            return Generation('USE_CASE | For kneading dough',10,8,.1,{})
    backend=Backend();extractor=HybridExtractor(backend,schema,'fixture prompt','v2-fixture')
    text='Capacity 2 liters. For kneading dough.';path=tmp_path/'v2-fixture'/'entry.json'
    first,reused=extractor.cached(path,'P1',text,{'details':{'Volume':'2 liters'}},13)
    assert not reused and first['status']=='ok' and len(first['claims'])==2
    second,reused=extractor.cached(path,'P1',text,{},13)
    assert reused and backend.calls==1 and second==first
    with pytest.raises(ValueError):extractor.cached(path,'P2',text,{},13)


def test_span_deduplication_is_case_whitespace_insensitive_with_fixed_type_priority():
    text='Recommended by testers. Better for baking.';audit={}
    claims,errors=parse_non_spec('SUBJECTIVE | recommended by testers\nUSE_CASE | Recommended  by testers\n'
        'AUTHORITY | Recommended by testers\nCOMPARATIVE | Better for baking\nTEMPORAL | Better for baking',text,audit=audit)
    assert not errors
    assert [c['claim_type'] for c in claims]==['AUTHORITY','COMPARATIVE']
    assert audit['duplicates_removed']['non_spec']==3
    assert len(audit['type_conflicts'])==3
    assert audit['type_conflicts'][1]['kept_type']=='AUTHORITY'


def test_truncated_word_is_rejected_and_actual_source_bytes_are_retained():
    text='Perfectly suitable for baking.'
    claims,errors=parse_non_spec('USE_CASE | suitable for bak\nSUBJECTIVE | Perfect\nUSE_CASE | suitable for BAKING',text)
    assert len(errors)==2 and all('not_in_source' in e for e in errors)
    assert len(claims)==1 and claims[0]['span']=='suitable for baking'


def test_specs_remove_non_product_short_values_and_duplicate_brand(schema):
    extended=deepcopy(schema)
    extended['attributes'] += [dict(canonical_name=n,aliases=[n.title()],value_type='categorical',variant_sensitive=False)
        for n in ('brand','manufacturer','package information','item model number','part number')]
    catalog={'details':{'Brand':'GreenBoxx','Manufacturer':'GREENBOXX','Material':'Red',
        'Package Information':'Box','Item Model Number':'ABC123','Part Number':'XYZ789'}}
    text='GreenBoxx GREENBOXX; Red red; Box ABC123 XYZ789; infrared.'
    audit={};claims=deterministic_specs(text,catalog,extended,audit)
    assert [(c['attribute'],c['span']) for c in claims]==[('brand','GreenBoxx'),('material','Red')]
    assert audit['duplicates_removed']['spec']==2
    assert deterministic_specs('Material PP; Weight 2g; infrared.',{'details':{'Material':'PP'}},schema)==[]


def test_counts_map_to_distinct_schema_attributes_and_mismatch_is_preserved(schema):
    extended=deepcopy(schema)
    extended['attributes'] += [dict(canonical_name=n,aliases=a,value_type='numeric+unit',variant_sensitive=True)
        for n,a in [('number of pieces',['Pieces']),('number of items',['Set','Pack']),('unit count',['Count'])]]
    catalog={'details':{'Pieces':'8','Set':'4','Count':'3 count'}}
    claims=deterministic_specs('16 Pieces. Set of 4. 2-Pack. 3 count.',catalog,extended)
    assert [(c['attribute'],c['span']) for c in claims]==[
        ('number of pieces','16 Pieces'),('number of items','Set of 4'),('number of items','2-Pack'),('unit count','3 count')]
    assert not any(c['unmapped'] for c in claims)
    assert check_catalog('number of pieces','16 Pieces',catalog,extended).status=='contradiction'
    assert check_catalog('number of items','Set of 4',catalog,extended).status=='support'


def test_equivalent_spec_values_are_deduplicated_and_dangling_spans_rejected(schema):
    extended=deepcopy(schema)
    extended['attributes'].append(dict(canonical_name='number of pieces',aliases=['Pieces'],value_type='numeric+unit',variant_sensitive=True))
    audit={};claims=deterministic_specs('16 Pcs. 16 Pieces.',{'details':{'Pieces':'16'}},extended,audit)
    assert len(claims)==1 and audit['duplicates_removed']['spec_values']==1
    claims,errors=parse_non_spec('AUTHORITY | Arranged by a young entrepreneur driven by a','Arranged by a young entrepreneur driven by a passion for nature.')
    assert not claims and errors==['line_0:truncated_phrase']


@pytest.mark.parametrize('phrase',['Set of 4-6','4-6 pieces','Set of 4.5','2.5 pieces'])
def test_count_ranges_and_fractions_never_become_supported_partial_counts(schema,phrase):
    extended=deepcopy(schema)
    extended['attributes'].append(dict(canonical_name='number of pieces',aliases=['Pieces'],value_type='numeric+unit',variant_sensitive=True))
    claims=deterministic_specs(phrase+'.',{'details':{'Pieces':'6'}},extended)
    assert len(claims)==1 and claims[0]['span']==phrase
    assert claims[0]['unmapped']
    assert check_catalog(claims[0]['attribute'],claims[0]['value'],{'details':{'Pieces':'6'}},extended).status!='support'
