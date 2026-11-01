"""CPU skeleton: pinned DeBERTa cross-encoder probabilities; no GPU fallback.

The  config labels are contradiction, entailment, neutral. Model inference is
lazy, CPU-only, eval/inference_mode, with explicit four-thread limits.
"""
from dataclasses import dataclass
import time

MODEL_ID='cross-encoder/nli-deberta-v3-base'
REVISION='6c749ce3425cd33b46d187e45b92bbf96ee12ec7'
LABELS=('contradiction','entailment','neutral')

@dataclass(frozen=True)
class NLIScores:
    contradiction:float
    entailment:float
    neutral:float

def scores_from_rows(rows):
    rows=list(rows)
    if any(len(row)!=3 for row in rows):raise ValueError('Expected three C/E/N probabilities')
    return [NLIScores(*map(float,row)) for row in rows]

class CPUCrossEncoderNLI:
    def __init__(self,model_path,*,model_id=MODEL_ID,revision=REVISION,batch_size=4,max_length=256,threads=4,device='cpu'):
        if device!='cpu' or not 1<=threads<=4 or batch_size<1 or max_length<1:raise ValueError('This adapter is CPU-only with at most four threads')
        import torch
        from transformers import AutoTokenizer,AutoModelForSequenceClassification
        torch.set_num_threads(threads)
        try:torch.set_num_interop_threads(1)
        except RuntimeError:
            if torch.get_num_interop_threads()>4:raise
        torch.manual_seed(13)
        self.torch=torch;self.batch_size=batch_size;self.max_length=max_length;self.model_id=model_id;self.revision=revision
        self.tokenizer=AutoTokenizer.from_pretrained(str(model_path),local_files_only=True)
        self.model=AutoModelForSequenceClassification.from_pretrained(str(model_path),local_files_only=True,torch_dtype=torch.float32).to('cpu').eval()
        mapping=tuple(str(self.model.config.id2label[i]).casefold() for i in range(3))
        if mapping!=LABELS:raise ValueError('NLI label mapping differs from the pinned T1 model')
        if any(p.device.type!='cpu' for p in self.model.parameters()):raise ValueError('Non-CPU parameter placement')
        self.last_timing={}
    def score(self,pairs):
        pairs=list(pairs);out=[];started=time.perf_counter();token_lengths=[]
        with self.torch.inference_mode():
            for offset in range(0,len(pairs),self.batch_size):
                batch=pairs[offset:offset+self.batch_size]
                tokens=self.tokenizer([p[0] for p in batch],[p[1] for p in batch],padding=True,truncation=True,max_length=self.max_length,return_tensors='pt')
                token_lengths.extend(tokens['attention_mask'].sum(dim=1).tolist())
                if any(v.device.type!='cpu' for v in tokens.values()):raise ValueError('Non-CPU inputs')
                logits=self.model(**tokens).logits
                out.extend(scores_from_rows(self.torch.softmax(logits,dim=-1).cpu().tolist()))
        if len(out)!=len(pairs):raise ValueError('NLI result count mismatch')
        elapsed=time.perf_counter()-started
        self.last_timing=dict(pairs=len(pairs),elapsed_s=elapsed,pairs_per_second=len(pairs)/elapsed if pairs else None,token_lengths=token_lengths,device='cpu',threads=self.torch.get_num_threads(),interop_threads=self.torch.get_num_interop_threads(),batch_size=self.batch_size,max_length=self.max_length,model_id=self.model_id,revision=self.revision,label_order=list(LABELS))
        return out
    def close(self):
        self.model=None;self.tokenizer=None
        import gc;gc.collect()
