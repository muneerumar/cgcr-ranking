"""CPU skeleton: read-only per-item BM25 review retrieval with provenance."""
from pathlib import Path
import re
from cgcr.data.review_sentences import SentenceIndex

class ReviewRetriever:
    def __init__(self,index_root):self.index_root=Path(index_root)
    def retrieve(self,item_id,claim,k=20):
        if not re.fullmatch(r'[A-Za-z0-9_-]+',item_id):raise ValueError('Unsafe item ID')
        path=self.index_root/(item_id+'.sqlite')
        if not path.is_file():return []
        with SentenceIndex(path) as index:return index.retrieve(claim,k)
