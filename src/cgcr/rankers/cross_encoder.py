"""pinned MS MARCO MiniLM cross-encoder via existing transformers (no extra deps)."""
import time
from cgcr.rankers.base import Ranker, RankResult

MODEL_ID='cross-encoder/ms-marco-MiniLM-L6-v2'
MODEL_REVISION='233902d25c440f23af6f7d6e94d2946bac0bee0a'


class CrossEncoderRanker(Ranker):
    def __init__(self, model_id=MODEL_ID, revision=MODEL_REVISION, device='cpu', batch_size=16, max_length=512, cache_dir=None):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.torch=torch
        self.tokenizer=AutoTokenizer.from_pretrained(model_id,revision=revision,cache_dir=cache_dir)
        self.model=AutoModelForSequenceClassification.from_pretrained(model_id,revision=revision,cache_dir=cache_dir,use_safetensors=True).to(device).eval()
        self.device,self.batch_size,self.max_length=device,batch_size,max_length
        self.model_id,self.revision=model_id,revision

    def rank(self,query,docs,seed):
        start=time.perf_counter();scores=[];tokens=0
        with self.torch.inference_mode():
            for offset in range(0,len(docs),self.batch_size):
                batch=docs[offset:offset+self.batch_size]
                inputs=self.tokenizer([query]*len(batch),batch,padding=True,truncation=True,max_length=self.max_length,return_tensors='pt').to(self.device)
                tokens+=int(inputs['attention_mask'].sum().item())
                scores.extend(self.model(**inputs).logits.reshape(-1).cpu().tolist())
        permutation=sorted(range(len(docs)),key=lambda i:(-scores[i],i))
        return RankResult(permutation,'','not_applicable',tokens,0,time.perf_counter()-start,
                          backend_stats={'scores':scores,'model_id':self.model_id,'revision':self.revision,
                                         'device':self.device,'max_pair_tokens':self.max_length})
