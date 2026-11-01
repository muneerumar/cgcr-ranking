"""extraction schema, private cache identity, provenance and annotation tests."""
import importlib.util
import json
from pathlib import Path
import pytest
from cgcr.defenses.cgcr.extract import ClaimEnvelope,Extractor,validate_claims,cache_key
from cgcr.defenses.cgcr.mlx_json import JSONTokenMask
from cgcr.defenses.cgcr.cohort import inserted_intervals,claim_origin
from cgcr.rankers.llm_listwise import Generation
SCHEMA={'attributes':[{'canonical_name':'item weight','aliases':['Weight'],'value_type':'numeric+unit','variant_sensitive':True}]}
def claim(**kw):return dict(attribute='Weight',value='2',unit='pounds',claim_type='SPEC',span='2 pounds',**kw)
def test_schema_maps_and_retains_unmapped():
    c=claim();out=validate_claims(json.dumps({'claims':[c]}),'Weight: 2 pounds',SCHEMA)
    assert out[0]['attribute']=='item weight' and not out[0]['unmapped']
    c['attribute']='novel property';assert validate_claims(json.dumps({'claims':[c]}),'2 pounds',SCHEMA)[0]['unmapped']
@pytest.mark.parametrize('change',[{'claim_type':'FAKE'},{'extra':1},{'unit':1},{'span':'not present'}])
def test_reject_schema_and_nonverbatim(change):
    c=claim();c.update(change)
    with pytest.raises(ValueError):validate_claims(json.dumps({'claims':[c]}),'2 pounds',SCHEMA)
class Backend:
    def __init__(self,outputs):self.outputs=iter(outputs);self.calls=[]
    def generate_json(self,messages,**kw):
        self.calls.append(kw);return Generation(next(self.outputs),10,10,.1,{})
def test_one_repair_and_terminal_failure_cached(tmp_path):
    backend=Backend(['bad','bad']);e=Extractor(backend,SCHEMA,'{attributes}','repair','v1')
    v,cached=e.cached(tmp_path/'x.json','item','2 pounds',13)
    assert v['status']=='extraction_failed' and len(v['attempts'])==2 and not cached
    v,cached=e.cached(tmp_path/'x.json','item','2 pounds',13);assert cached and len(backend.calls)==2
    assert cache_key('item','2 pounds','v1')!=cache_key('item','2 pounds','v2')
def test_repair_success_and_json_constraints_forwarded():
    b=Backend(['bad',json.dumps({'claims':[claim()]})]);v=Extractor(b,SCHEMA,'{attributes}','repair','v').extract('i','2 pounds',1)
    assert v['status']=='repaired' and b.calls[0]['json_schema']==ClaimEnvelope.model_json_schema()
def test_token_mask_discards_prefill_remainder():
    class E:
        def __init__(self):self.seen=[]
        def get_allowed_tokens(self,tokens):
            self.seen.append(tokens);return type('L',(),{'allowed_tokens':[1]})()
    e=E();mask=JSONTokenMask(e);mask.allowed([9,8]);mask.allowed([9,8,7]);assert e.seen==[[],[7]]
def test_inserted_span_provenance():
    edits=[{'start':3,'end':3,'replacement':'xyz'}];assert inserted_intervals(edits)==[(3,6)]
    assert claim_origin('xyz','abcxyz',[(3,6)])=='inserted'
    assert claim_origin('abc','abcxyz',[(3,6)])=='clean'
    assert claim_origin('a','aba',[(2,3)])=='ambiguous'

def test_partial_failure_preserves_valid_claims_and_logs_rejected():
    good=claim();bad=claim();bad['span']='invented span'
    raw=json.dumps({'claims':[good,bad]});backend=Backend([raw,raw])
    v=Extractor(backend,SCHEMA,'{attributes}','repair {errors}','v1').extract('item','2 pounds',13)
    assert v['status']=='partial_failure' and len(v['claims'])==1
    assert len(v['rejected_claims'])==1 and len(v['attempts'])==2
