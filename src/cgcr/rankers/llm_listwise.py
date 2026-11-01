"""listwise ranking with audited parsing, one repair, deterministic final fill."""
from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path
import re
import time
from urllib.request import Request, urlopen

from cgcr.rankers.base import Ranker, RankResult


@dataclass
class Generation:
    text: str
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_s: float
    stats: dict = field(default_factory=dict)


@dataclass
class ParsedRanking:
    indices: list[int]
    valid: bool
    reason: str


def parse_ranking(text, n):
    """Find a contiguous ranking while ignoring unrelated IDs in surrounding prose."""
    patterns = [r'\[\s*-?\d+(?:\s*,\s*-?\d+)+\s*\]',
                r'\[\s*-?\d+\s*\](?:\s*(?:>|,|→|->)?\s*\[\s*-?\d+\s*\])*',
                r'(?<!\d)-?\d+(?:\s*(?:,|>|→|->)\s*-?\d+)+']
    candidates=[]
    for pattern in patterns:
        for match in re.finditer(pattern,text):
            candidates.append((match.start(),list(map(int,re.findall(r'-?\d+',match.group())))))
    if re.fullmatch(r'\s*-?\d+(?:\s+-?\d+)*\s*',text):
        candidates.append((0,list(map(int,text.split()))))
    complete=[(pos,values) for pos,values in candidates if len(values)==n and sorted(values)==list(range(1,n+1))]
    if complete:
        numbers=max(complete,key=lambda pair:pair[0])[1]
        ambiguous=len({tuple(values) for _,values in complete})>1
        return ParsedRanking([v-1 for v in numbers],not ambiguous,'ambiguous' if ambiguous else 'ok')
    numbers=max(candidates,key=lambda pair:(len(pair[1]),pair[0]))[1] if candidates else []
    indices=list(dict.fromkeys(v-1 for v in numbers if 1<=v<=n))
    reason='duplicate' if len(set(numbers))<len(numbers) else 'out_of_range' if any(v<1 or v>n for v in numbers) else 'incomplete'
    return ParsedRanking(indices,False,reason)


class ListwiseRanker(Ranker):
    def __init__(self, backend, prompt_path, repair_prompt_path, max_tokens=128, temperature=0., log_raw_output=True):
        self.backend = backend
        self.template = Path(prompt_path).read_text()
        self.repair_template = Path(repair_prompt_path).read_text()
        self.max_tokens, self.temperature = max_tokens, temperature
        self.log_raw_output = log_raw_output

    def rank(self, query, docs, seed):
        start = time.perf_counter()
        block = '\n\n'.join(f'[{i+1}] {doc}' for i,doc in enumerate(docs))
        prompt = self.template.format(query=query,candidates_block=block,n=len(docs))
        messages = [{'role':'user','content':prompt}]
        attempts=[]
        for attempt in range(2):
            output = self.backend.generate(messages, seed=seed, max_tokens=self.max_tokens, temperature=self.temperature)
            parsed = parse_ranking(output.text,len(docs))
            attempts.append(asdict(output)|{'parse_valid':parsed.valid,'parse_reason':parsed.reason,'parsed_indices':parsed.indices})
            if self.log_raw_output:
                logging.getLogger('cgcr').info('Ranking attempt %s valid=%s reason=%s raw=%r',attempt,parsed.valid,parsed.reason,output.text)
            else:
                logging.getLogger('cgcr').info('Ranking attempt %s valid=%s reason=%s',attempt,parsed.valid,parsed.reason)
            if parsed.valid:
                status='ok' if attempt==0 else 'repaired'
                break
            if attempt==0:
                messages += [{'role':'\u0061ssistant','content':output.text},
                             {'role':'user','content':self.repair_template.format(n=len(docs))}]
        else:
            status='repaired_fill'
        permutation = parsed.indices + [i for i in range(len(docs)) if i not in parsed.indices]
        total = lambda field: sum(a[field] for a in attempts) if all(a[field] is not None for a in attempts) else None
        return RankResult(permutation,output.text,status,total('prompt_tokens'),total('completion_tokens'),
                          time.perf_counter()-start,attempts,{'attempt_count':len(attempts)})


class OllamaBackend:
    """debug-only local HTTP adapter; no cloud endpoint or silent fallback."""
    def __init__(self, model, base_url='http://127.0.0.1:11434', timeout=600, num_ctx=8192):
        self.model,self.base_url,self.timeout,self.num_ctx=model,base_url.rstrip('/'),timeout,num_ctx

    def generate(self, messages, *, seed, max_tokens, temperature):
        body={'model':self.model,'messages':messages,'stream':False,'keep_alive':'10m',
              'options':{'seed':seed,'temperature':temperature,'num_predict':max_tokens,'num_ctx':self.num_ctx}}
        start=time.perf_counter()
        request=Request(self.base_url+'/api/chat',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        with urlopen(request,timeout=self.timeout) as response:
            data=json.load(response)
        stats={k:data.get(k) for k in ('model','done_reason','total_duration','load_duration','prompt_eval_duration','eval_duration')}
        for target,tokens,duration in [('prompt_tokens_per_s','prompt_eval_count','prompt_eval_duration'),('generation_tokens_per_s','eval_count','eval_duration')]:
            if data.get(duration,0)>0 and data.get(tokens) is not None:
                stats[target]=data[tokens]/(data[duration]*1e-9)
        return Generation(data['message']['content'],data.get('prompt_eval_count'),data.get('eval_count'),time.perf_counter()-start,stats)


    def close(self):
        """Unload the debug LLM before this client process exits."""
        body={'model':self.model,'keep_alive':0}
        request=Request(self.base_url+'/api/generate',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        with urlopen(request,timeout=self.timeout) as response:
            response.read()


class MLXBackend:
    """primary Mac adapter; lazy mlx-lm import, explicit model path, greedy sampling."""
    def __init__(self, model_path):
        from cgcr.rankers.memory import ensure_available
        ensure_available()
        from mlx_lm import load
        self.model,self.tokenizer=load(str(model_path))
        self.model_path=str(model_path)

    def generate(self,messages,*,seed,max_tokens,temperature):
        import mlx.core as mx
        from cgcr.rankers.memory import ensure_available,memory_snapshot
        ensure_available()
        resident_peak=memory_snapshot()['resident_bytes']
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        mx.random.seed(seed)
        mx.reset_peak_memory()
        prompt=self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        start=time.perf_counter();response=None;text=[]
        for response in stream_generate(self.model,self.tokenizer,prompt=prompt,max_tokens=max_tokens,sampler=make_sampler(temp=temperature)):
            ensure_available()
            resident_peak=max(resident_peak,memory_snapshot()['resident_bytes'])
            text.append(response.text)
        mx.synchronize()
        if response is None:
            return Generation('',0,0,time.perf_counter()-start,{'peak_memory_bytes':mx.get_peak_memory()})
        return Generation(''.join(text),response.prompt_tokens,response.generation_tokens,time.perf_counter()-start,
                          {'prompt_tokens_per_s':response.prompt_tps,'generation_tokens_per_s':response.generation_tps,
                           'peak_memory_bytes':mx.get_peak_memory(),'peak_resident_bytes':resident_peak,
                           'process_lifetime_peak_resident_bytes':__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss,'model_path':self.model_path})


    def close(self):
        """Release this process's only LLM before another can be loaded."""
        import gc
        import mlx.core as mx
        self.model=None
        self.tokenizer=None
        gc.collect()
        mx.clear_cache()
