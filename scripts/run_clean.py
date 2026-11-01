#!/usr/bin/env python3
"""traceable dev-only clean ranking smoke harness; no final/test evaluation."""
import argparse
from collections import Counter,defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import pyarrow.parquet as pq
from cgcr.metrics.ranking import ndcg_at_k,recall_exact_at_k,kendall_tau,DEFAULT_GAINS
from cgcr.rankers.inputs import render_input,INPUT_MODES
from cgcr.rankers.permutation import rank_with_permutations
from cgcr.utils.io import read_yaml,write_jsonl,write_yaml
from cgcr.utils.logging import start_run,get_logger

ROOT=Path(__file__).resolve().parents[1]


def save_json(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def load_pilot(root,limit=None):
    out=Path(root)/'data/processed/Home_and_Kitchen'
    ids=json.loads((out/'pilot_query_ids.json').read_text())['query_ids']
    splits=json.loads((out/'split_ids.json').read_text())['query_ids']
    if not set(ids)<=set(splits['dev']) or set(ids)&set(splits['test']):
        raise ValueError('Pilot must be dev-only')
    ids=sorted(ids)[:limit] if limit else sorted(ids)
    selected=set(ids);groups=defaultdict(list)
    # Read only dev data; test parquet is never opened.
    for row in pq.read_table(out/'dev.parquet').to_pylist():
        if row['query_id'] in selected:groups[row['query_id']].append(row)
    if set(groups)!=selected or any(len(rows)!=10 for rows in groups.values()):
        raise ValueError('Pilot needs ten candidates per query')
    return groups


def make_ranker(args,config,root=ROOT):
    if args.ranker=='bm25':
        from cgcr.rankers.bm25 import BM25Ranker
        return BM25Ranker()
    if args.ranker=='cross_encoder':
        from cgcr.rankers.cross_encoder import CrossEncoderRanker
        import torch
        torch.set_num_threads(config['ranker']['cross_encoder']['cpu_threads'])
        return CrossEncoderRanker(**config['ranker']['cross_encoder']['init'],cache_dir=str(root/'models/cache'))
    from cgcr.rankers.llm_listwise import ListwiseRanker,MLXBackend,OllamaBackend
    if args.backend=='mlx':
        model=config['models'][args.model_key]
        backend=MLXBackend(root/model['path'])
    elif args.backend=='ollama':
        backend=OllamaBackend(config['ranker']['ollama']['model'],num_ctx=config['ranker']['max_context_tokens'])
    else:
        raise ValueError('Choose the Mac or local HTTP backend')
    return ListwiseRanker(backend,root/config['ranker']['prompt'],root/config['ranker']['repair_prompt'],
                          max_tokens=config['ranker']['max_new_tokens'],temperature=config['ranker']['temperature'])


def source_snapshot(run,root):
    root=ROOT  # snapshot the executing implementation, even for fixture artifact roots
    paths=[p for folder in ('src/cgcr/rankers','src/cgcr/metrics','src/cgcr/data','src/cgcr/utils','configs/prompts','configs/schemas')
           for p in (root/folder).glob('*') if p.suffix in {'.py','.txt','.yaml'}]
    paths += [root/'scripts'/name for name in ('run_clean.py','convert_models.py')]
    paths += [root/'configs/cgcr_default.yaml']
    manifest={}
    for path in paths:
        relative=path.relative_to(root);destination=run/'source'/relative
        destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(path.read_bytes())
        manifest[str(relative)]=hashlib.sha256(path.read_bytes()).hexdigest()
    inputs={}
    candidates=[root/'data/processed/Home_and_Kitchen'/name for name in ('dev.parquet','pilot_query_ids.json','split_ids.json')]
    candidates += list((root/'models').glob('*/cgcr_provenance.json'))
    for path in candidates:
        if path.is_file():
            digest=hashlib.sha256()
            with path.open('rb') as stream:
                while chunk:=stream.read(1024*1024):digest.update(chunk)
            inputs[str(path.relative_to(root))]=digest.hexdigest()
    save_json(run/'input_manifest.json',inputs)
    save_json(run/'code_manifest.json',manifest)
    save_json(run/'config_hash.json',{'sha256':hashlib.sha256((run/'config.yaml').read_bytes()).hexdigest()})


def evaluate(root,config,ranker,groups,*,ranker_name,backend,input_mode,orders=3,model_key=None,phase='smoke'):
    root=Path(root);schema=read_yaml(root/'configs/schemas/Home_and_Kitchen.yaml')
    resolved=config|{'execution':{'ranker':ranker_name,'backend':backend,'input_mode':input_mode,'orders':orders,
                                  'model_key':model_key,'phase':phase,'queries':sorted(groups),'split':'dev','paper_result':False}}
    run=start_run(f'T4_smoke_{ranker_name}_{backend}_{input_mode}_{phase}',resolved,root/'results')
    source_snapshot(run,root);write_yaml(run/'seeds.yaml',{'seed':config['seed']})
    logger=get_logger();outputs=[];started=time.monotonic()
    with (run/'per_query.jsonl').open('w') as stream:
        for index,(qid,rows) in enumerate(sorted(groups.items())):
            seed=int.from_bytes(hashlib.sha256(f"{config['seed']}\0{qid}".encode()).digest()[:4],'big')
            rendered=[render_input(r,input_mode,schema,config['ranker']['max_doc_chars']) for r in rows]
            try:
                result=rank_with_permutations(ranker,rows[0]['query'],[t for t,_ in rendered],seed,orders)
            except BaseException as exc:
                failure={'status':'aborted' if isinstance(exc,(KeyboardInterrupt,MemoryError)) else 'failed',
                         'error':type(exc).__name__+': '+str(exc),'completed_queries':len(outputs),'query_id':qid}
                if backend=='mlx':
                    from cgcr.rankers.memory import memory_snapshot
                    import mlx.core as mx
                    failure.update(memory_snapshot());failure['peak_metal_bytes']=mx.get_peak_memory()
                save_json(run/'failure.json',failure);save_json(run/'metrics.json',failure)
                logger.exception('Smoke run interrupted')
                raise
            labels=[r['esci_label'] for r in rows];eligible=any(label!='E' for label in labels)
            if any(r['utility_eligible']!=eligible for r in rows):raise ValueError('Utility flag mismatch')
            record={'query_id':qid,'item_ids':[r['item_id'] for r in rows],'labels':labels,'utility_eligible':eligible,
                    'input_mode':input_mode,'truncated_docs':sum(t for _,t in rendered),**asdict(result),
                    'ndcg_at_10':ndcg_at_k(labels,result.permutation,10,config['metrics']['gains'],utility_eligible=eligible),
                    'recall_exact_at_5':recall_exact_at_k(labels,result.permutation,5),
                    'kendall_tau_to_input':kendall_tau(list(range(len(rows))),result.permutation)}
            stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');stream.flush();outputs.append(record)
            if index%10==0:logger.info('Smoke query %s/%s complete',index+1,len(groups))
    results=[o['result'] for r in outputs for o in r['orders']]
    attempts=[a for r in results for a in r['attempts']]
    ndcg=[r['ndcg_at_10'] for r in outputs if r['ndcg_at_10'] is not None]
    average=lambda values:statistics.mean(values) if values else None
    count=lambda key:sum(r[key] for r in results) if all(r[key] is not None for r in results) else None
    resident_lifetime_peaks=[a['stats']['process_lifetime_peak_resident_bytes'] for a in attempts if 'process_lifetime_peak_resident_bytes' in a['stats']]
    resident_peaks=[a['stats']['peak_resident_bytes'] for a in attempts if 'peak_resident_bytes' in a['stats']]
    gpu_peaks=[a['stats']['peak_memory_bytes'] for a in attempts if 'peak_memory_bytes' in a['stats']]
    throughput={}
    for field,tokens in [('prompt_tokens_per_s','prompt_tokens'),('generation_tokens_per_s','completion_tokens')]:
        measured=[a for a in attempts if a['stats'].get(field,0)>0]
        elapsed=sum(a[tokens]/a['stats'][field] for a in measured)
        throughput[field]=sum(a[tokens] for a in measured)/elapsed if elapsed else None
    latencies=[r['latency_s'] for r in results]
    metrics={'label':'smoke_NOT_FOR_PAPER','ranker':ranker_name,'backend':backend,'model_key':model_key,'input_mode':input_mode,'phase':phase,
             'queries':len(outputs),'utility_eligible_queries':len(ndcg),'ndcg_at_10':average(ndcg),
             'recall_exact_at_5':average([r['recall_exact_at_5'] for r in outputs if r['recall_exact_at_5'] is not None]),
             'kendall_tau_to_input':average([r['kendall_tau_to_input'] for r in outputs if r['kendall_tau_to_input'] is not None]),
             'ranking_calls':len(results),'generation_attempts':len(attempts),'parse_status_counts':dict(Counter(r['parse_status'] for r in results)),
             'initial_parse_failure_rate':sum(not a['parse_valid'] for r in results for a in r['attempts'][:1])/len(results) if attempts else None,
             'final_fill_rate':sum(r['parse_status']=='repaired_fill' for r in results)/len(results) if attempts else None,
             'prompt_tokens':count('prompt_tokens'),'completion_tokens':count('completion_tokens'),
             'seconds_per_ranking_call':average(latencies),'ranking_latency_p50_s':statistics.median(latencies),
             'ranking_latency_max_s':max(latencies),'elapsed_s':time.monotonic()-started,
             'peak_process_lifetime_resident_bytes':max(resident_lifetime_peaks) if resident_lifetime_peaks else None,
             'peak_resident_memory_bytes':max(resident_peaks) if resident_peaks else None,
             'peak_mlx_memory_bytes':max(gpu_peaks) if gpu_peaks else None,'truncated_documents':sum(r['truncated_docs'] for r in outputs),**throughput}
    save_json(run/'metrics.json',metrics)
    logger.info('Smoke complete: %s',run.name)
    print(json.dumps({'run':str(run.relative_to(root)),'metrics':metrics},indent=2),flush=True)
    return run,metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split',choices=['dev'],default='dev')
    parser.add_argument('--subset',choices=['pilot'],default='pilot')
    parser.add_argument('--ranker',choices=['bm25','cross_encoder','llm'],required=True)
    parser.add_argument('--backend',choices=['mlx','ollama'],default='mlx')
    parser.add_argument('--input',choices=INPUT_MODES,default='seller_text')
    parser.add_argument('--orders',type=int,default=3)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--model-key',default='qwen_4bit')
    parser.add_argument('--config',default='configs/cgcr_default.yaml')
    args=parser.parse_args();config=read_yaml(ROOT/args.config)
    from cgcr.utils.seeding import set_seed
    set_seed(config['seed'])
    ranker=make_ranker(args,config)
    try:
        evaluate(ROOT,config,ranker,load_pilot(ROOT,args.limit),ranker_name=args.ranker,
                 backend=args.backend if args.ranker=='llm' else 'local',input_mode=args.input,
                 orders=args.orders,model_key=(args.model_key if args.backend=='mlx' else config['ranker']['ollama']['model'] if args.backend=='ollama' else config['ranker']['model']) if args.ranker=='llm' else None)
    finally:
        backend=getattr(ranker,'backend',None)
        if hasattr(backend,'close'):backend.close()


if __name__=='__main__':main()
