"""parsing, repair/fill, aggregation and network-free backend/harness checks."""
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys

import pytest
from cgcr.rankers.base import RankResult,validate_permutation
from cgcr.rankers.llm_listwise import Generation,ListwiseRanker,parse_ranking,OllamaBackend
from cgcr.rankers.permutation import aggregate_rankings,rank_with_permutations
from cgcr.rankers.bm25 import BM25Ranker
from cgcr.rankers.inputs import render_input
from cgcr.utils.io import read_yaml

ROOT=Path(__file__).parents[1]


@pytest.mark.parametrize('text,expected,valid',[
    ('3,1,2',[2,0,1],True),('[3] > [1] > [2]',[2,0,1],True),
    ('In 2026 the order is [3, 1, 2].',[2,0,1],True),
    ('Here is the ranking: 3 > 1 > 2. Thank you.',[2,0,1],True),
    ('3\n1\n2',[2,0,1],True),('[3] > [1]',[2,0],False),
    ('3,3,1',[2,0],False),('4,1,2',[0,1],False),
    ('none',[],False),('0,1,2',[0,1],False),
])
def test_parser(text,expected,valid):
    result=parse_ranking(text,3)
    assert result.indices==expected and result.valid==valid


class MockBackend:
    def __init__(self,outputs):self.outputs=iter(outputs);self.requests=[]
    def generate(self,messages,**kwargs):
        self.requests.append((list(messages),kwargs))
        return Generation(next(self.outputs),11,3,.01)


@pytest.mark.parametrize('outputs,status,expected',[
    (['3,1,2'],'ok',[2,0,1]),(['3,1','2,3,1'],'repaired',[1,2,0]),
    (['3,3,1','[2] > [2]'],'repaired_fill',[1,0,2]),(['bad','bad'],'repaired_fill',[0,1,2])])
def test_one_repair_then_fill(tmp_path,outputs,status,expected):
    backend=MockBackend(outputs)
    ranker=ListwiseRanker(backend,ROOT/'configs/prompts/rank_listwise.txt',ROOT/'configs/prompts/rank_repair.txt')
    result=ranker.rank('query',['a','b','c'],7)
    assert result.permutation==expected and result.parse_status==status
    assert len(backend.requests)==len(outputs)<=2
    assert result.prompt_tokens==11*len(outputs)
    assert result.completion_tokens==3*len(outputs)
    assert '[1] a' in backend.requests[0][0][0]['content']
    assert all(request[1]['temperature']==0 for request in backend.requests)


def test_aggregation_median_mean_index_ties_and_shuffles():
    order,medians,means=aggregate_rankings([[0,1,2],[2,1,0],[1,0,2]],3)
    assert medians==[2,2,3] and order==[1,0,2]  # same median; item1 lower mean
    assert aggregate_rankings([[0,1],[1,0]],2)[0]==[0,1]  # mean also tied
    class Identity:
        def rank(self,q,docs,seed):return RankResult(list(range(len(docs))),'','ok',0,0,0)
    first=rank_with_permutations(Identity(),'q',['a','b','c'],13,3)
    assert first==rank_with_permutations(Identity(),'q',['a','b','c'],13,3)
    assert all(o['permutation_original']==o['shuffle'] for o in first.orders)
    with pytest.raises(ValueError):validate_permutation([0,0],2)


def test_bm25_empty_and_relevant_docs():
    ranker=BM25Ranker()
    assert ranker.rank('coffee',['coffee maker','blue fabric','green chair','white sheet'],13).permutation[0]==0
    assert ranker.rank('q',['',''],13).permutation==[0,1]


def test_ollama_http_usage(monkeypatch):
    import cgcr.rankers.llm_listwise as module
    class Response:
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def read(self):return json.dumps({'message':{'content':'2,1'},'prompt_eval_count':9,'eval_count':3}).encode()
    seen=[]
    def opener(req,**kwargs):seen.append(json.loads(req.data));return Response()
    monkeypatch.setattr(module,'urlopen',opener)
    result=OllamaBackend('debug').generate([{'role':'user','content':'q'}],seed=13,max_tokens=32,temperature=0)
    assert result.text=='2,1' and result.prompt_tokens==9 and result.completion_tokens==3
    assert seen[0]['options']['seed']==13 and not seen[0]['stream']
    monkeypatch.setattr(sys,'platform','darwin')


def test_evidence_input_excludes_prose_and_text_attributes():
    schema=read_yaml(ROOT/'configs/schemas/Home_and_Kitchen.yaml')
    row={'seller_text':'Seller baseline','catalog':{'title':'Seller title','features':['prose'],
          'details':json.dumps({'Brand':'Brand','Color':'Red','Special Feature':'Unlicensed prose'}),'categories':['Home','Cups']}}
    text,truncated=render_input(row,'neutral_v0_details_only',schema)
    assert 'Brand | Cups' in text and 'color:' in text
    assert 'Seller title' not in text and 'Unlicensed prose' not in text
    assert render_input(row,'seller_text',schema)[0]=='Seller baseline'
    assert render_input(row,'seller_text',schema,4)==('Sell',True)


def test_mocked_complete_harness_no_network(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('run_clean_test',ROOT/'scripts/run_clean.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    schema_dir=tmp_path/'configs/schemas';schema_dir.mkdir(parents=True)
    (schema_dir/'Home_and_Kitchen.yaml').write_bytes((ROOT/'configs/schemas/Home_and_Kitchen.yaml').read_bytes())
    config=read_yaml(ROOT/'configs/cgcr_default.yaml')
    rows=[{'query_id':'q','query':'query','item_id':str(i),'esci_label':'E' if i==0 else 'S',
           'utility_eligible':True,'seller_text':f'Doc {i}','catalog':{'details':'{}','categories':['Cups']}} for i in range(10)]
    backend=MockBackend([','.join(map(str,range(1,11)))]*3)
    ranker=ListwiseRanker(backend,ROOT/'configs/prompts/rank_listwise.txt',ROOT/'configs/prompts/rank_repair.txt')
    monkeypatch.setattr(module,'source_snapshot',lambda run,root:None)
    run,metrics=module.evaluate(tmp_path,config,ranker,{'q':rows},ranker_name='llm',backend='mock',input_mode='seller_text',orders=3)
    assert metrics['ranking_calls']==3 and metrics['prompt_tokens']==33 and metrics['utility_eligible_queries']==1
    assert metrics['initial_parse_failure_rate']==0
    record=json.loads((run/'per_query.jsonl').read_text())
    assert len(record['median_ranks'])==10 and len(record['orders'])==3
    assert metrics['label']=='smoke_NOT_FOR_PAPER'


def test_cross_encoder_pinned_loading_and_scoring(monkeypatch):
    import torch
    import transformers
    from types import SimpleNamespace
    from cgcr.rankers.cross_encoder import CrossEncoderRanker,MODEL_ID,MODEL_REVISION
    calls=[]
    class Tokens(dict):
        def to(self,device):return self
    class Tokenizer:
        def __call__(self,queries,docs,**kwargs):
            return Tokens(input_ids=torch.tensor([[int(d),1] for d in docs]),attention_mask=torch.ones((len(docs),2),dtype=torch.long))
    class Model:
        def to(self,device):return self
        def eval(self):return self
        def __call__(self,**inputs):return SimpleNamespace(logits=inputs['input_ids'][:,:1].float())
    def tokenizer_loader(name,**kwargs):calls.append((name,kwargs['revision']));return Tokenizer()
    def model_loader(name,**kwargs):calls.append((name,kwargs['revision']));return Model()
    monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',tokenizer_loader)
    monkeypatch.setattr(transformers.AutoModelForSequenceClassification,'from_pretrained',model_loader)
    result=CrossEncoderRanker().rank('q',['1','3','2'],13)
    assert result.permutation==[1,2,0] and result.prompt_tokens==6
    assert calls==[(MODEL_ID,MODEL_REVISION)]*2


def test_mlx_generation_usage_and_greedy_seed(monkeypatch):
    import types
    from cgcr.rankers.llm_listwise import MLXBackend
    import cgcr.rankers.memory as memory
    seeds=[];samplers=[]
    mx=types.ModuleType('mlx.core')
    mx.random=types.SimpleNamespace(seed=seeds.append)
    mx.reset_peak_memory=lambda:None;mx.synchronize=lambda:None;mx.get_peak_memory=lambda:123
    parent=types.ModuleType('mlx');parent.core=mx
    mlx_lm=types.ModuleType('mlx_lm')
    tokenizer=types.SimpleNamespace(apply_chat_template=lambda *a,**k:'rendered prompt')
    mlx_lm.load=lambda path:(object(),tokenizer)
    mlx_lm.stream_generate=lambda *a,**kw:iter([types.SimpleNamespace(text='2,1',prompt_tokens=10,generation_tokens=3,prompt_tps=100.,generation_tps=20.)])
    sampler=types.ModuleType('mlx_lm.sample_utils');sampler.make_sampler=lambda temp:samplers.append(temp)
    for name,value in [('mlx',parent),('mlx.core',mx),('mlx_lm',mlx_lm),('mlx_lm.sample_utils',sampler)]:monkeypatch.setitem(sys.modules,name,value)
    monkeypatch.setattr(memory,'ensure_available',lambda:None)
    monkeypatch.setattr(memory,'memory_snapshot',lambda:{'resident_bytes':10,'available_bytes':2_000_000_000})
    output=MLXBackend('fake-model').generate([{'role':'user','content':'q'}],seed=7,max_tokens=32,temperature=0)
    assert output.text=='2,1' and output.prompt_tokens==10 and output.completion_tokens==3
    assert seeds==[7] and samplers==[0]
    assert output.stats['peak_memory_bytes']==123 and output.stats['peak_resident_bytes']==10


@pytest.mark.parametrize('text',[
    'Order [3] > [1] > [2]. Explanation: [1] matches the query.',
    'Order: 3,1,2. Candidate [1] has useful specifications.',
])
def test_stray_explanatory_ids_do_not_corrupt_ranking(text):
    result=parse_ranking(text,3)
    assert result.valid and result.indices==[2,0,1]



def test_negative_ids_are_not_silently_positive():
    assert not parse_ranking('-1,2,3',3).valid
    assert not parse_ranking('[-1,2,3]',3).valid
    assert parse_ranking('[3] -> [1] -> [2]',3).indices==[2,0,1]
    assert not parse_ranking('[3,1,2] or [2,1,3]',3).valid
