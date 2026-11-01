"""reimplementation: GPT-2-small CPU perplexity filter, dev 5% FPR cutoff.

Sliding token windows score each next token once. Flagged documents become a
fixed empty-evidence marker, preserving all ten candidate slots. This is a
generic PPL baseline, not a claimed reproduction of a paper-specific threshold.
"""
import math


def dev_threshold(scores, fpr=.05):
    import numpy as np
    if not scores or not 0 <= fpr < 1 or any(not math.isfinite(x) for x in scores):
        raise ValueError('Finite clean scores and valid FPR required')
    # Strict greater-than plus higher quantile keeps empirical FPR <= requested.
    return float(np.quantile(scores, 1-fpr, method='higher'))


def filter_document(text, score, threshold):
    if not math.isfinite(score):
        raise ValueError('Cannot silently filter an unscored document')
    return '[Document withheld by perplexity filter.]' if score > threshold else text


class GPT2Perplexity:
    def __init__(self, model_path, window=256, stride=128):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not 1 <= stride < window <= 1024:
            raise ValueError('Invalid GPT-2 sliding window')
        torch.set_num_threads(4)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            if torch.get_num_interop_threads() > 4:raise
        self.torch=torch;self.window=window;self.stride=stride
        self.tokenizer=AutoTokenizer.from_pretrained(str(model_path),local_files_only=True)
        self.model=AutoModelForCausalLM.from_pretrained(str(model_path),local_files_only=True,
                    torch_dtype=torch.float32,attn_implementation='eager').to('cpu').eval()

    def score(self, text):
        ids=self.tokenizer.encode(text,add_special_tokens=False)
        if len(ids)<2:return dict(log_perplexity=0.,perplexity=1.,tokens_scored=0)
        # Window endpoints advance by stride; overlap is context only, never re-scored.
        total=0.;count=0;previous_end=1
        with self.torch.inference_mode():
            for end in range(min(self.window,len(ids)),len(ids)+self.stride,self.stride):
                end=min(end,len(ids));start=max(0,end-self.window)
                tokens=self.torch.tensor([ids[start:end]],device='cpu')
                labels=tokens.clone();labels[:,:max(1,previous_end-start)]=-100
                scored=int((labels[:,1:]!=-100).sum())
                if scored:
                    loss=self.model(input_ids=tokens,labels=labels).loss
                    total+=float(loss)*scored;count+=scored
                previous_end=end
                if end==len(ids):break
        if count!=len(ids)-1:raise ValueError('PPL token accounting mismatch')
        logp=total/count
        return dict(log_perplexity=logp,perplexity=math.exp(min(logp,700)),tokens_scored=count)

    def close(self):
        self.model=None;self.tokenizer=None
        import gc;gc.collect()
