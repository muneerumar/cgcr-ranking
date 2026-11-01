"""numeric/fragment guards, coarse labels, NONE grammar and paired benchmark."""
import importlib
from pathlib import Path
from types import SimpleNamespace
import pytest
from cgcr.defenses.cgcr.claim_quality import malformed_numeric,fragment_flags,quality_summary,coarse
from cgcr.defenses.cgcr.hybrid_v3 import HybridV3Extractor,non_spec_reason
from cgcr.rankers.llm_listwise import Generation

@pytest.mark.parametrize('value', ['0 x 8\'','12 x 0 inches','12 x'])
def test_malformed_dimensions(value):assert malformed_numeric(value)

@pytest.mark.parametrize('value',['2.29 pounds','1.42 Liters','6.5"D x 13.5"W x 6.5"H','Box 2','GreenBoxx 16 Pieces'])
def test_valid_numeric_or_categorical_value(value):assert not malformed_numeric(value)


def test_non_spec_guards_are_not_semantic_accuracy_claims():
    for kind in ['USE_CASE','AUTHORITY','TEMPORAL']:
        assert non_spec_reason(kind,'GreenBoxx Kitchen')
        assert non_spec_reason(kind,'16 Pieces')
    assert non_spec_reason('USE_CASE','suitable for storing dry pasta') is None
    assert non_spec_reason('AUTHORITY','recommended by product testers') is None
    assert non_spec_reason('TEMPORAL','newly released this year') is None
    assert non_spec_reason('SUBJECTIVE','Beautiful')=='fewer_than_two_words'
    assert coarse('SPEC')==coarse('USE_CASE')=='CHECKABLE'
    assert coarse('AUTHORITY')=='NON_LICENSABLE'


def test_quality_reports_by_type_and_keeps_valid_one_word_spec_diagnostic():
    values=[dict(claims=[dict(claim_type='SPEC',value='Blue',span='Blue'),dict(claim_type='USE_CASE',value='12 oz',span='12 oz')])]
    q=quality_summary(values)
    assert q['per_type']['SPEC']['fragment_rate']==1.
    assert q['per_type']['USE_CASE']['mostly_digits_units']==1
    assert q['coarse_counts']['CHECKABLE']==2 and 'not semantic accuracy' in q['caveat']


def test_v3_keeps_none_and_rejects_bad_numeric_specs():
    class Backend:
        def generate(self,*a,**k):return Generation('NONE',5,1,.01,{})
    ext=HybridV3Extractor(Backend(),{'attributes':[]},'fixture','v3-fixture')
    value=ext.extract('P','Broken dimensions 0 x 8\'.',{'details':{}},13)
    assert value['status']=='ok' and not value['claims']
    assert any(r['reason']=='malformed_numeric' for r in value['quality_rejections'])


def test_v3_grammar_removes_single_word_and_bad_type_paths():
    from cgcr.defenses.cgcr.mlx_v3 import V3LineConstraint
    class Tokenizer:
        eos_token_ids=[9999]
        def encode(self,s,**kw):return list(s.encode())
    grammar=V3LineConstraint(Tokenizer(),['Beautiful','GreenBoxx Kitchen','suitable for storing dry pasta'])
    # All candidate paths for a one-word phrase are removed; NONE remains.
    assert 0 in grammar.nodes[0]['remaining']
    assert all(1+i*3 not in grammar.nodes[0]['remaining'] for i in range(5))
    assert 1+0*3+1 not in grammar.nodes[0]['remaining'] # name as USE_CASE
    assert 1+0*3+2 in grammar.nodes[0]['remaining']


