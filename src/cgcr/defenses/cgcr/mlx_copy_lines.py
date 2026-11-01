"""v2: small finite-state source-copy grammar for non-SPEC short lines.

This enforces format and verbatim provenance, not semantic correctness. Phrase
choices are complete punctuation-delimited source clauses of at most 200 chars.
The 15-line/256-token limits remain; EOS is reserved before the hard token limit.
"""
import re
import time
from cgcr.rankers.llm_listwise import MLXBackend,Generation
from cgcr.defenses.cgcr.hybrid_extract import complete_phrase,normalized_span

TYPES=('USE_CASE','COMPARATIVE','AUTHORITY','TEMPORAL','SUBJECTIVE')


def source_phrases(text):
    result=[]
    for clause in re.finditer(r'[^\n.!?;:,]+',text):
        phrase=clause.group().strip(' \t\r\n•')
        # Never force an arbitrary eight-word prefix into a purported atomic
        # claim. Long clauses stay unavailable rather than being cut mid-thought.
        if phrase and len(phrase)<=200 and complete_phrase(phrase):result.append(phrase)
    unique={}
    for phrase in result:unique.setdefault(normalized_span(phrase),phrase)
    return list(unique.values())


class CopyLineConstraint:
    def __init__(self,tokenizer,phrases,max_lines=15,max_generated=252):
        self.max_lines,self.max_generated=max_lines,max_generated
        self.eos=set(tokenizer.eos_token_ids)
        self.prefill=None
        self.nodes=[]
        def node():
            self.nodes.append(dict(children={},ends=set(),remaining={}))
            return len(self.nodes)-1
        node()
        lines=['NONE']+[kind+' | '+phrase+'\n' for kind in TYPES for phrase in phrases]
        self.same_span={0:{0}}
        for index in range(len(phrases)):
            group={1+kind*len(phrases)+index for kind in range(len(TYPES))}
            for identity in group:self.same_span[identity]=group
        for identity,line in enumerate(lines):
            tokens=tokenizer.encode(line,add_special_tokens=False);current=0
            for position in range(len(tokens)+1):
                self.nodes[current]['remaining'][identity]=len(tokens)-position
                if position==len(tokens):self.nodes[current]['ends'].add(identity);break
                token=tokens[position]
                if token not in self.nodes[current]['children']:self.nodes[current]['children'][token]=node()
                current=self.nodes[current]['children'][token]

    def allowed(self,tokens):
        tokens=list(tokens)
        if self.prefill is None:self.prefill=len(tokens)
        generated=tokens[self.prefill:];current=0;used=set();lines=0;none=False
        for token in generated:
            ends=self.nodes[current]['ends']
            if ends:
                if 0 in ends:none=True
                else:
                    for identity in ends:used.update(self.same_span[identity])
                    lines+=1;current=0
            if token in self.eos:return sorted(self.eos)
            if none or token not in self.nodes[current]['children']:raise ValueError('Source-copy grammar state mismatch')
            current=self.nodes[current]['children'][token]
        ends=self.nodes[current]['ends']
        if ends:
            if 0 in ends:return sorted(self.eos)
            for identity in ends:used.update(self.same_span[identity])
            lines+=1;current=0
        if lines>=self.max_lines:return sorted(self.eos)
        allowed=set(self.eos) if current==0 and lines else set()
        budget=self.max_generated-len(generated)
        for token,child in self.nodes[current]['children'].items():
            if any(identity not in used and (identity!=0 or lines==0) and remaining+2<=budget
                   for identity,remaining in self.nodes[child]['remaining'].items()):allowed.add(token)
        if not allowed:raise ValueError('Source-copy grammar has no completion within the token budget')
        return sorted(allowed)

    def __call__(self,tokens,logits):
        import numpy as np
        import mlx.core as mx
        allowed=self.allowed(tokens.tolist())
        mask=np.full(logits.shape[-1],-float('inf'),dtype=np.float32);mask[allowed]=0.
        return logits+mx.array(mask)


class CopyLineMLXBackend(MLXBackend):
    def generate(self,messages,*,seed,max_tokens,temperature):
        if temperature!=0 or max_tokens!=256:raise ValueError('v2 short-line settings must remain temperature 0 / 256 tokens')
        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        from cgcr.rankers.memory import ensure_available,memory_snapshot
        ensure_available();mx.random.seed(seed);mx.reset_peak_memory()
        started=time.perf_counter();phrases=source_phrases(messages[-1]['content'])
        processor=CopyLineConstraint(self.tokenizer,phrases)
        prompt=self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        response=None;text=[];resident=memory_snapshot()['resident_bytes']
        for response in stream_generate(self.model,self.tokenizer,prompt=prompt,max_tokens=256,
                sampler=make_sampler(temp=0.),logits_processors=[processor],prefill_step_size=512):
            ensure_available();resident=max(resident,memory_snapshot()['resident_bytes']);text.append(response.text)
        mx.synchronize()
        return Generation(''.join(text),response.prompt_tokens if response else 0,response.generation_tokens if response else 0,
            time.perf_counter()-started,dict(peak_metal_bytes=mx.get_peak_memory(),peak_resident_bytes=resident,
            source_copy_constraint=True,source_phrase_candidates=len(phrases),source_phrase_boundaries='whole punctuation clauses',
            max_phrase_chars=200,span_unique_decoding=True,
            max_lines=15,semantic_correctness_enforced=False,temperature=0.,model_path=self.model_path))
