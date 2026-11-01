"""deterministic BM25Okapi reference using Unicode word tokenization."""
import re
import time
from cgcr.rankers.base import Ranker, RankResult


def tokenize(text):
    return re.findall(r'\w+', text.casefold())


class BM25Ranker(Ranker):
    def rank(self, query, docs, seed):
        from rank_bm25 import BM25Okapi
        start = time.perf_counter()
        corpus = [tokenize(doc) for doc in docs]
        scores = BM25Okapi(corpus).get_scores(tokenize(query)).tolist() if any(corpus) else [0.] * len(docs)
        permutation = sorted(range(len(docs)), key=lambda i: (-scores[i], i))
        return RankResult(permutation, '', 'not_applicable', 0, 0, time.perf_counter()-start,
                          backend_stats={'scores': scores, 'tokens_not_applicable': True})
