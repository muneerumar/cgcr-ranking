"""sanitizer speed adaptation: official FramingBench prompt, per-document cache.

Reimplementation of the runner, not an unchanged official baseline. The MIT
prompt at 8a53ba8f99853fb3220b419f7ad107dea69a814b is unchanged. Each input is
one ## Candidate 1 section; joint cross-product equalization is unavailable.
Qwen 4-bit replaces upstream Llama-70B. Cap/malformed outputs retain the input
and are flagged. Rewrites and raw outputs are stored only in private/.
"""
from dataclasses import asdict
from pathlib import Path
from cgcr.defenses.baselines.sanitize import comparison_document,parse_sanitized
from cgcr.defenses.cgcr.cohort import load,save
from cgcr.defenses.cgcr.extract import sha
from cgcr.rankers.fixed_order import stable_seed


def output_budget(input_tokens):
    if input_tokens<1:raise ValueError('Input tokens must be positive')
    return min(512,max(1,input_tokens*13//10))


class DocumentSanitizer:
    def __init__(self,backend_factory,cache,prompt,model,seed=13,version='sanitizer_document_cache_v2'):
        self.backend_factory=backend_factory;self.cache=Path(cache);self.prompt=prompt
        self.model=model;self.seed=seed;self.version=version

    def sanitize(self,qid,text):
        identity=sha(dict(query_id=str(qid),text=text,prompt=self.prompt,model=self.model,
                          version=self.version,seed=self.seed,cap='min(512,floor(1.3*document_tokens))'))
        path=self.cache/(identity+'.json');reused=path.exists()
        if reused:record=load(path)
        else:
            backend=self.backend_factory();document=comparison_document([text])
            tokens=len(backend.tokenizer.encode(document,add_special_tokens=False));cap=output_budget(tokens)
            output=backend.generate([{'role':'system','content':self.prompt},{'role':'user','content':document}],
                seed=stable_seed(self.seed,str(qid),sha(text),'T7-sanitize-document-v2'),max_tokens=cap,temperature=0.)
            if output.completion_tokens is None:raise ValueError('Sanitizer output token count missing')
            hit=output.completion_tokens>=cap;parsed=parse_sanitized(output.text,1)
            valid=parsed is not None and not hit
            record=dict(version=self.version,raw=asdict(output),text=parsed[0] if valid else text,input_tokens=tokens,
                max_new_tokens=cap,hit_cap=hit,status='ok' if valid else 'fallback_original_token_cap' if hit else 'fallback_original_bad_sections')
            save(path,record)
        raw=record['raw']
        safe=dict(cache_key=identity,cache_reused=reused,status=record['status'],hit_cap=record['hit_cap'],
            input_tokens=record['input_tokens'],max_new_tokens=record['max_new_tokens'],
            prompt_tokens=raw['prompt_tokens'],completion_tokens=raw['completion_tokens'],
            generation_seconds=raw['latency_s'],new_generation_seconds=0. if reused else raw['latency_s'],
            stats=raw['stats'])
        return record['text'],safe
