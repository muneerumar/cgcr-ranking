"""shared single-model v2/v3 benchmark backend and filtered source-copy grammar."""
import time
from cgcr.defenses.cgcr.mlx_copy_lines import CopyLineMLXBackend,CopyLineConstraint,source_phrases,TYPES
from cgcr.defenses.cgcr.hybrid_v3 import non_spec_reason
from cgcr.rankers.llm_listwise import Generation


class V3LineConstraint(CopyLineConstraint):
    def __init__(self,tokenizer,phrases,**kwargs):
        super().__init__(tokenizer,phrases,**kwargs)
        valid={0}|{1+i*len(phrases)+j for i,kind in enumerate(TYPES) for j,phrase in enumerate(phrases) if non_spec_reason(kind,phrase) is None}
        for node in self.nodes:
            node['remaining']={identity:n for identity,n in node['remaining'].items() if identity in valid}
            node['ends'].intersection_update(valid)


class V3CopyBackend(CopyLineMLXBackend):
    mode='v3'
    def generate(self,messages,*,seed,max_tokens,temperature):
        if self.mode=='v2':return super().generate(messages,seed=seed,max_tokens=max_tokens,temperature=temperature)
        if temperature!=0 or max_tokens!=256:raise ValueError('Expected greedy 256-token extraction')
        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        from cgcr.rankers.memory import ensure_available,memory_snapshot
        ensure_available();mx.random.seed(seed);mx.reset_peak_memory();started=time.perf_counter()
        phrases=source_phrases(messages[-1]['content']);processor=V3LineConstraint(self.tokenizer,phrases)
        prompt=self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        response=None;text=[];resident=memory_snapshot()['resident_bytes']
        for response in stream_generate(self.model,self.tokenizer,prompt=prompt,max_tokens=256,
                sampler=make_sampler(temp=0.),logits_processors=[processor],prefill_step_size=512):
            ensure_available();resident=max(resident,memory_snapshot()['resident_bytes']);text.append(response.text)
        mx.synchronize()
        return Generation(''.join(text),response.prompt_tokens if response else 0,response.generation_tokens if response else 0,
            time.perf_counter()-started,dict(peak_metal_bytes=mx.get_peak_memory(),peak_resident_bytes=resident,
                source_copy_constraint=True,quality_filtered_types=True,semantic_correctness_enforced=False,model_path=self.model_path))
