"""MLX greedy decoding constrained token by token with LM Format Enforcer."""
import time
from cgcr.rankers.llm_listwise import MLXBackend,Generation

class JSONTokenMask:
    """MLX first processor input is the prefill remainder, not the entire prompt."""
    def __init__(self,enforcer):self.enforcer=enforcer;self.prefill_tokens=None
    def allowed(self,tokens):
        tokens=list(tokens)
        if self.prefill_tokens is None:self.prefill_tokens=len(tokens)
        return self.enforcer.get_allowed_tokens(tokens[self.prefill_tokens:]).allowed_tokens
    def __call__(self,tokens,logits):
        import numpy as np
        import mlx.core as mx
        allowed=self.allowed(tokens.tolist())
        if not allowed:raise RuntimeError('Constrained decoder has no valid tokens')
        mask=np.full(logits.shape[-1],-float('inf'),dtype=np.float32);mask[allowed]=0.
        return logits+mx.array(mask)

class JSONMLXBackend(MLXBackend):
    def __init__(self,model_path):
        super().__init__(model_path)
        from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data
        tokenizer=getattr(self.tokenizer,'_tokenizer',self.tokenizer)
        self.enforcer_data=build_token_enforcer_tokenizer_data(tokenizer,use_bitmask=False)
    def generate_json(self,messages,*,json_schema,seed,max_tokens):
        from lmformatenforcer import JsonSchemaParser,TokenEnforcer
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        import mlx.core as mx
        from cgcr.rankers.memory import ensure_available,memory_snapshot
        ensure_available();mx.random.seed(seed);mx.reset_peak_memory()
        processor=JSONTokenMask(TokenEnforcer(self.enforcer_data,JsonSchemaParser(json_schema)))
        prompt=self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        started=time.perf_counter();texts=[];response=None;resident=memory_snapshot()['resident_bytes']
        for response in stream_generate(self.model,self.tokenizer,prompt=prompt,max_tokens=max_tokens,
                sampler=make_sampler(temp=0.),logits_processors=[processor],prefill_step_size=512):
            ensure_available();resident=max(resident,memory_snapshot()['resident_bytes']);texts.append(response.text)
        mx.synchronize()
        return Generation(''.join(texts),response.prompt_tokens if response else 0,response.generation_tokens if response else 0,
            time.perf_counter()-started,dict(peak_metal_bytes=mx.get_peak_memory(),peak_resident_bytes=resident,
            prompt_tokens_per_s=response.prompt_tps if response else None,generation_tokens_per_s=response.generation_tps if response else None,
            constrained_json=True,temperature=0.,model_path=self.model_path))
