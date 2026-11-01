"""forked shared-prefix MLX KV state, with exact token-prefix validation.

The chunk boundary matches uncached prefill to reduce numerical differences.
Only a target-last shared prefix is prepared by the caller. Every completion
forks the immutable prefix; repairs fall back unless the exact prefix matches.
"""
import time
from cgcr.rankers.llm_listwise import MLXBackend, Generation
from cgcr.rankers.memory import ensure_available, memory_snapshot


def common_prefix_length(left, right):
    for i, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return i
    return min(len(left), len(right))


class CachedMLXBackend(MLXBackend):
    def __init__(self, model_path, prefill_step_size=512):
        super().__init__(model_path)
        self.prefill_step_size = prefill_step_size
        self.clear_prefix()

    def clear_prefix(self):
        self.prefix_tokens = []
        self.prefix_cache = None
        self.prefix_setup_s = 0.
        self.prefix_peak_metal_bytes = 0
        self.prefix_peak_resident_bytes = 0

    def tokens(self, messages):
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        bos = self.tokenizer.bos_token
        return self.tokenizer.encode(prompt, add_special_tokens=bos is None or not prompt.startswith(bos))

    def prepare_prefix(self, user_prefix, full_messages):
        """Cache whole prefill chunks, strictly before the target body.

        Derive the prefix from the actual rendered chat string (no assumed chat
        delimiters). The full user message must begin with user_prefix.
        """
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache
        self.clear_prefix()
        if len(full_messages) != 1 or full_messages[0]['role'] != 'user':
            raise ValueError('Prefix preparation requires the initial user prompt')
        content = full_messages[0]['content']
        if not content.startswith(user_prefix):
            raise ValueError('Prefix does not match the full user message')
        full = self.tokenizer.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=True)
        # Locate content from the right: template control text cannot cause an
        # accidental early match when an empty prefix or duplicate query occurs.
        start = full.rfind(content)
        if start < 0:
            raise ValueError('Chat template did not preserve the user content')
        prefix_string = full[:start+len(user_prefix)]
        bos = self.tokenizer.bos_token
        p = self.tokenizer.encode(prefix_string, add_special_tokens=bos is None or not prefix_string.startswith(bos))
        all_tokens = self.tokens(full_messages)
        length = min(common_prefix_length(p, all_tokens), len(all_tokens)-1)
        length = length//self.prefill_step_size*self.prefill_step_size
        if length == 0:
            return 0.
        cache = make_prompt_cache(self.model)
        started = time.perf_counter()
        mx.reset_peak_memory()
        peak_rss=ensure_available()['resident_bytes']
        for offset in range(0, length, self.prefill_step_size):
            peak_rss=max(peak_rss,ensure_available()['resident_bytes'])
            self.model(mx.array(all_tokens[offset:offset+self.prefill_step_size])[None], cache=cache)
            mx.eval([c.state for c in cache])
            mx.clear_cache()
        mx.synchronize()
        self.prefix_peak_metal_bytes=mx.get_peak_memory()
        self.prefix_peak_resident_bytes=max(peak_rss,ensure_available()['resident_bytes'])
        self.prefix_tokens, self.prefix_cache = all_tokens[:length], cache
        self.prefix_setup_s = time.perf_counter()-started
        return self.prefix_setup_s

    def _fork(self):
        import mlx.core as mx
        # Explicit copies prevent a later KV update from touching the shared
        # prefix. MLX cache state consists of arrays, paired for keys/values.
        fork = [type(c).from_state(tuple(__import__('copy').deepcopy(x) for x in c.state), c.meta_state)
                for c in self.prefix_cache]
        mx.eval([c.state for c in fork])
        return fork

    def generate(self, messages, *, seed, max_tokens, temperature):
        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        ensure_available()
        tokens = self.tokens(messages)
        reuse = len(self.prefix_tokens) if self.prefix_cache is not None and tokens[:len(self.prefix_tokens)] == self.prefix_tokens else 0
        started = time.perf_counter()
        cache = self._fork() if reuse else None
        mx.random.seed(seed)
        mx.reset_peak_memory()
        peak_rss = memory_snapshot()['resident_bytes']
        response = None
        texts = []
        for response in stream_generate(self.model, self.tokenizer, prompt=tokens[reuse:],
                max_tokens=max_tokens, sampler=make_sampler(temp=temperature),
                prompt_cache=cache, prefill_step_size=self.prefill_step_size):
            peak_rss = max(peak_rss, ensure_available()['resident_bytes'])
            texts.append(response.text)
        mx.synchronize()
        elapsed = time.perf_counter()-started
        stats = {'logical_prompt_tokens': len(tokens), 'reused_prefix_tokens': reuse,
                 'processed_prompt_tokens': len(tokens)-reuse, 'prefix_setup_s': self.prefix_setup_s,
                 'peak_memory_bytes': max(mx.get_peak_memory(),self.prefix_peak_metal_bytes),
                 'peak_resident_bytes': max(peak_rss,self.prefix_peak_resident_bytes),
                 'prompt_tokens_per_s': response.prompt_tps if response else None,
                 'generation_tokens_per_s': response.generation_tps if response else None}
        del cache
        mx.clear_cache()
        return Generation(''.join(texts), len(tokens), response.generation_tokens if response else 0, elapsed, stats)

    def close(self):
        self.clear_prefix()
        super().close()
