#!/usr/bin/env python3
"""convert one pinned official model per process using mlx_lm.convert."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

from cgcr.rankers.memory import ensure_available,memory_snapshot
from cgcr.utils.io import read_yaml,write_yaml,write_jsonl
from cgcr.utils.logging import start_run,get_logger

ROOT=Path(__file__).resolve().parents[1]



def stage_source(model, source_cache, max_attempts=6):
    """Retry interrupted HTTP snapshots; the Hub keeps its cache/prefixes."""
    from huggingface_hub import snapshot_download
    import logging
    for attempt in range(1,max_attempts+1):
        try:
            return snapshot_download(model['source_repo'],revision=model['source_revision'],cache_dir=source_cache,
                allow_patterns=['*.json','*.safetensors','*.jinja','*.txt','tokenizer.model','*.tiktoken','README.md','LICENSE*'],max_workers=4)
        except Exception as exc:
            status=getattr(getattr(exc,'response',None),'status_code',None)
            if status in {401,404} or attempt==max_attempts:
                raise
            logging.getLogger('cgcr').warning('Source download attempt %s/%s failed (%s); retaining cached partials',attempt,max_attempts,type(exc).__name__)
            time.sleep(min(2**attempt,30))



def verify_source_weights(source):
    """Compare complete source shards to their official SHA-256 cache addresses."""
    import re
    files={}
    for path in sorted(Path(source).glob('*.safetensors')):
        expected=path.resolve().name
        if not re.fullmatch(r'[0-9a-f]{64}',expected):
            raise ValueError('Expected a Hub SHA-256 blob address for source '+path.name)
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            while chunk:=stream.read(1024*1024):digest.update(chunk)
        actual=digest.hexdigest()
        if actual!=expected:raise ValueError('Source shard checksum mismatch: '+path.name)
        files[path.name]={'bytes':path.stat().st_size,'sha256':actual}
    if not files:raise ValueError('No official safetensors source shards found')
    return files


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-key',choices=['qwen_4bit','llama_4bit'],required=True)
    parser.add_argument('--source-cache',default=str(ROOT/'models/source-cache'))
    args=parser.parse_args();config=read_yaml(ROOT/'configs/cgcr_default.yaml');model=config['models'][args.model_key]
    out=ROOT/model['path']
    if out.exists():raise FileExistsError(f'Refusing to overwrite {out}')
    run=start_run('T4_smoke_convert_'+args.model_key,config|{'conversion':model,'source_cache':args.source_cache},ROOT/'results')
    write_yaml(run/'seeds.yaml',{'seed':config['seed']});start=time.monotonic()
    source=stage_source(model,args.source_cache)
    source_weights=verify_source_weights(source)
    (run/'source_weight_hashes.json').write_text(json.dumps(source_weights,indent=2)+'\n')
    source_config=json.loads((Path(source)/'config.json').read_text())
    (run/'source_model_config.json').write_text(json.dumps(source_config,indent=2)+'\n')
    (run/'convert_source.py').write_bytes(Path(__file__).read_bytes())
    ensure_available()
    import mlx.core as mx
    from mlx_lm.convert import convert
    from cgcr.utils.seeding import set_seed
    set_seed(config['seed'])
    mx.random.seed(config['seed'])
    mx.reset_peak_memory()
    try:
        convert(hf_path=source,mlx_path=str(out),quantize=True,q_group_size=model['quantization_group_size'],q_bits=model['quantization_bits'],q_mode=model['quantization_mode'])
        mx.synchronize();ensure_available()
        actual=json.loads((out/'config.json').read_text())['quantization']
        assert actual['bits']==model['quantization_bits'] and actual['group_size']==model['quantization_group_size']
        files={}
        for path in sorted(out.iterdir()):
            if path.is_file():
                h=hashlib.sha256()
                with path.open('rb') as stream:
                    while chunk:=stream.read(1024*1024):h.update(chunk)
                files[path.name]={'bytes':path.stat().st_size,'sha256':h.hexdigest()}
        metrics={'status':'completed','model_key':args.model_key,'source_repo':model['source_repo'],
                 'source_revision':model['source_revision'],'source_weights':source_weights,'quantization':actual,'files':files,
                 'total_bytes':sum(v['bytes'] for v in files.values()),'elapsed_s':time.monotonic()-start,
                 'peak_metal_bytes':mx.get_peak_memory(),**memory_snapshot()}
        (out/'cgcr_provenance.json').write_text(json.dumps(metrics,indent=2)+'\n')
        (run/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
        write_jsonl(run/'per_query.jsonl',[metrics])
        get_logger().info('Converted %s to %s',args.model_key,out)
    except BaseException as exc:
        failure={'status':'aborted' if isinstance(exc,(KeyboardInterrupt,MemoryError)) else 'failed','error':type(exc).__name__+': '+str(exc),'partial_output':str(out),'elapsed_s':time.monotonic()-start,'peak_metal_bytes':mx.get_peak_memory(),**memory_snapshot()}
        (run/'metrics.json').write_text(json.dumps(failure,indent=2)+'\n')
        write_jsonl(run/'per_query.jsonl',[failure]);get_logger().exception('Conversion aborted')
        raise


if __name__=='__main__':main()
