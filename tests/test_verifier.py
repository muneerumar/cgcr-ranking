""" CPU skeleton: unit normalization, evidence policy, robust estimators and CPU NLI."""
from dataclasses import asdict
from pathlib import Path
import importlib.util
import math
import sys
from types import SimpleNamespace
from contextlib import nullcontext
import pytest
from cgcr.defenses.cgcr.catalog import check_catalog
from cgcr.defenses.cgcr.aggregation import aggregate_reviews
from cgcr.defenses.cgcr.nli import CPUCrossEncoderNLI,NLIScores,scores_from_rows
from cgcr.defenses.cgcr.review_evidence import review_evidence
from cgcr.defenses.cgcr.retrieval import ReviewRetriever

def schema():
    return {'attributes':[dict(canonical_name=n,aliases=[n],value_type=t,variant_sensitive=v) for n,t,v in (
        ('capacity','numeric+unit',True),('product dimensions','numeric+unit',True),('color','categorical',True),('batteries required','boolean',False),('special feature','text',False))]}

def test_catalog_units_tolerance_variant_and_attackable_fields():
    catalog=dict(details={'capacity':'1 Liter','batteries required':True,'special feature':'Persuasion'},features=['capacity 2 liters'])
    assert check_catalog('capacity','1000 ml',catalog,schema()).status=='support'
    assert check_catalog('capacity','1.04 Liters',catalog,schema()).status=='support'
    assert check_catalog('capacity','2 Liters',catalog,schema()).status=='contradiction'
    assert check_catalog('capacity','2 Liters',catalog,schema(),joined_via_variant=True).status=='unknown_variant'
    assert check_catalog('capacity','1 Liter',catalog,schema(),joined_via_variant=True).status=='support'
    assert check_catalog('capacity','12-16 oz',catalog,schema()).status=='range_unparsed'
    assert check_catalog('batteries required','yes',catalog,schema()).status=='support'
    assert check_catalog('special feature','Persuasion',catalog,schema()).status=='excluded_attackable'
    assert check_catalog('absent','x',catalog,schema()).status=='unknown'
    with pytest.raises(ValueError):check_catalog('capacity','1 L',catalog,schema(),rel_tolerance=float('nan'))

def test_dimension_axes_are_aligned_without_guessing_unknown_axes():
    catalog={'details':{'product dimensions':'6.5"D x 13.5"W x 6.5"H'}}
    assert check_catalog('product dimensions','13.5"W x 6.5"D x 6.5"H',catalog,schema()).status=='support'
    assert check_catalog('product dimensions','6.5 x 13.5 x 6.5 inches',catalog,schema()).status=='unknown'

def row(e,c,user='u',review='r',stamp=0):
    return dict(entailment=e,contradiction=c,neutral=1-e-c,user_id=user,review_id=review,verified_purchase=True,helpful_vote=0,timestamp=stamp)

def test_reviewer_caps_bound_duplicate_user_contribution():
    rows=[row(1,0,'spam','a'),row(1,0,'spam','b'),row(0,1,'buyer','c')]
    naive=aggregate_reviews(rows,'mean');robust=aggregate_reviews(rows,'reviewer_capped')
    assert math.isclose(naive.support_strength,1/3)
    assert robust.support_strength==0 and robust.reviewer_weights=={'spam':1.,'buyer':1.}
    assert robust.effective_weight==2

def test_trimmed_mean_missing_users_windows_and_empty():
    rows=[row(0,1,'low')]+[row(.8,.1,str(i)) for i in range(8)]+[row(1,0,'high')]
    result=aggregate_reviews(rows,'trimmed_mean',trim_fraction=.1)
    assert result.evidence_count==8 and math.isclose(result.support_strength,.7)
    anonymous=[row(1,0,None,'same'),row(1,0,None,'same'),row(0,1,None,'other')]
    assert aggregate_reviews(anonymous).reviewer_weights=={'review:same':1.,'review:other':1.}
    capped=aggregate_reviews(rows,time_window_ms=1000,per_window_cap=2.)
    assert capped.effective_weight<=2.0000001
    assert aggregate_reviews([]).support_strength==0
    with pytest.raises(ValueError):aggregate_reviews([row(1,1)])
    with pytest.raises(ValueError):aggregate_reviews(rows,trim_fraction=.5)
    with pytest.raises(ValueError):aggregate_reviews(rows,time_window_ms=1,per_window_cap=float('nan'))

def test_type_priors_prevent_unsupported_authority_scoring():
    class Never:
        def retrieve(self,*args):raise AssertionError('must not retrieve')
        def score(self,*args):raise AssertionError('must not infer')
    assert review_evidence('P','dummy',Never(),Never(),claim_type='AUTHORITY')['status']=='requires_external'
    assert review_evidence('P','dummy',Never(),Never(),claim_type='SUBJECTIVE')['status']=='never'

def test_retriever_missing_item_and_traversal(tmp_path):
    retriever=ReviewRetriever(tmp_path)
    assert retriever.retrieve('P','capacity')==[]
    with pytest.raises(ValueError):retriever.retrieve('../P','capacity')

def test_nli_mapping_and_cpu_only_rejection():
    result=scores_from_rows([[.2,.7,.1]])[0]
    assert result.entailment==.7 and result.contradiction==.2 and result.neutral==.1
    with pytest.raises(ValueError):scores_from_rows([[1.,0.]])
    with pytest.raises(ValueError):CPUCrossEncoderNLI('unused',device='mps')

def test_mocked_cpu_adapter_explicit_threads_labels_and_scoring(monkeypatch):
    calls={}
    class Tensor:
        device=SimpleNamespace(type='cpu')
        def __init__(self,rows):self.rows=rows
        def sum(self,dim):return Tensor([len(x) for x in self.rows])
        def tolist(self):return self.rows
        def cpu(self):return self
    class Tokenizer:
        def __call__(self,a,b,**kwargs):return {'input_ids':Tensor([[1,2,3] for _ in a]),'attention_mask':Tensor([[1,1,1] for _ in a])}
    class Model:
        config=SimpleNamespace(id2label={0:'contradiction',1:'entailment',2:'neutral'})
        def to(self,device):calls['device']=device;return self
        def eval(self):return self
        def parameters(self):return [Tensor([])]
        def __call__(self,**kwargs):return SimpleNamespace(logits=Tensor([[.1,.8,.1] for _ in kwargs['input_ids'].rows]))
    torch=SimpleNamespace(float32='float32',set_num_threads=lambda n:calls.update(threads=n),set_num_interop_threads=lambda n:calls.update(interop=n),manual_seed=lambda s:None,
        get_num_threads=lambda:calls['threads'],get_num_interop_threads=lambda:calls['interop'],inference_mode=nullcontext,softmax=lambda t,dim:t)
    trans=SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a,**k:Tokenizer()),AutoModelForSequenceClassification=SimpleNamespace(from_pretrained=lambda *a,**k:Model()))
    monkeypatch.setitem(sys.modules,'torch',torch);monkeypatch.setitem(sys.modules,'transformers',trans)
    nli=CPUCrossEncoderNLI('fixture');out=nli.score([('premise','claim'),('premise2','claim2')])
    assert len(out)==2 and out[0].entailment==.8 and calls=={'threads':4,'interop':1,'device':'cpu'}
    assert nli.last_timing['pairs']==2 and nli.last_timing['label_order']==['contradiction','entailment','neutral']



def test_review_components_return_scores_and_evidence_trace():
    from cgcr.data.review_sentences import ReviewSentence
    class Retriever:
        def retrieve(self,item,claim,k):
            return [ReviewSentence(1,'P:0','text',0,'Fixture sentence','hashed-user',True,0,5.,1000,1.)]
    class NLI:
        model_id='fixture';revision='immutable-fixture'
        def score(self,pairs):
            assert pairs==[('Fixture sentence','Fixture claim')]
            return [NLIScores(.1,.8,.1)]
    out=review_evidence('P','Fixture claim',Retriever(),NLI())
    assert out['status']=='scored' and math.isclose(out['score'],.7)
    assert out['trace'][0]['review_id']=='P:0' and out['revision']=='immutable-fixture'


