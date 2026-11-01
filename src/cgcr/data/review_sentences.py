"""immutable, per-item SQLite FTS5 BM25 sentence indexes.

Review titles and bodies are split separately using deterministic punctuation/
newline rules, not a learned segmenter. Source reviewer hashes are preserved.
SQLite BM25 uses k1=1.2,b=.75 and returns lower scores first; the API negates
scores so larger relevance is better. See https://sqlite.org/fts5.html.
"""
from dataclasses import dataclass,asdict
import hashlib
import html
import json
from pathlib import Path
import re
import sqlite3
import time
from urllib.parse import quote
import pyarrow.parquet as pq

VERSION='sentence-fts5-v1'

def split_sentences(text):
    if not text:return []
    text=re.sub(r'<br\s*/?>','\n',str(text),flags=re.I)
    text=html.unescape(re.sub(r'<[^>]+>',' ',text))
    return [re.sub(r'\s+',' ',part).strip() for part in re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"“])|\n+',text) if part.strip()]

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block:=stream.read(1024*1024):h.update(block)
    return h.hexdigest()

@dataclass(frozen=True)
class ReviewSentence:
    sentence_id:int
    review_id:str
    field:str
    sentence_no:int
    text:str
    user_id:str|None
    verified_purchase:bool
    helpful_vote:int
    rating:float|None
    timestamp:int|None
    relevance:float

class SentenceIndex:
    def __init__(self,path):
        self.path=Path(path)
        self.db=sqlite3.connect('file:'+quote(str(self.path.resolve()))+'?mode=ro',uri=True)
        self.db.row_factory=sqlite3.Row;self.db.execute('PRAGMA cache_size=-1024')
    def metadata(self):return {r[0]:json.loads(r[1]) for r in self.db.execute('SELECT key,value FROM metadata')}
    def retrieve(self,query,k=20):
        if k<0:raise ValueError('k must be nonnegative')
        tokens=list(dict.fromkeys(re.findall(r'\w+',str(query).casefold())))[:64]
        if not tokens or not k:return []
        match=' OR '.join('"'+t.replace('"','""')+'"' for t in tokens)
        sql='SELECT s.*, bm25(sentence_fts) AS score FROM sentence_fts JOIN sentences s ON s.sentence_id=sentence_fts.rowid WHERE sentence_fts MATCH ? ORDER BY score,s.sentence_id LIMIT ?'
        return [ReviewSentence(**{key:r[key] for key in ('sentence_id','review_id','field','sentence_no','text','user_id','helpful_vote','rating','timestamp')},verified_purchase=bool(r['verified_purchase']),relevance=-r['score']) for r in self.db.execute(sql,(match,k))]
    def close(self):self.db.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()

def build_item_index(source,destination,item_id):
    source=Path(source);destination=Path(destination)
    if not re.fullmatch(r'[A-Za-z0-9_-]+',item_id):raise ValueError('Unsafe item ID')
    signature=dict(version=VERSION,source_bytes=source.stat().st_size,source_mtime_ns=source.stat().st_mtime_ns,item_id=item_id)
    if destination.exists():
        with SentenceIndex(destination) as index:metadata=index.metadata()
        if all(metadata.get(k)==v for k,v in signature.items()):
            return metadata|dict(index_bytes=destination.stat().st_size,reused=True)
        raise ValueError('Index/source signature changed; choose a new output version')
    started=time.monotonic();destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_name('.'+destination.name+'.building')
    for suffix in ('','-journal'):Path(str(temporary)+suffix).unlink(missing_ok=True)
    db=sqlite3.connect(temporary)
    reviews=sentences=0
    try:
        db.execute('PRAGMA cache_size=-2048');db.execute('PRAGMA temp_store=FILE');db.execute('PRAGMA synchronous=NORMAL')
        db.executescript('CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT); CREATE TABLE sentences(sentence_id INTEGER PRIMARY KEY,review_id TEXT,field TEXT,sentence_no INTEGER,text TEXT,user_id TEXT,verified_purchase INTEGER,helpful_vote INTEGER,rating REAL,timestamp INTEGER); CREATE VIRTUAL TABLE sentence_fts USING fts5(text,content=sentences,content_rowid=sentence_id,tokenize=unicode61);')
        for batch in pq.ParquetFile(source).iter_batches(batch_size=256,use_threads=False):
            pending=[]
            for review in batch.to_pylist():
                review_id=item_id+':'+str(reviews);reviews+=1
                for field in ('title','text'):
                    for number,sentence in enumerate(split_sentences(review.get(field))):
                        sentences+=1
                        pending.append((sentences,review_id,field,number,sentence,review.get('user_id'),int(bool(review.get('verified_purchase'))),int(review.get('helpful_vote') or 0),review.get('rating'),review.get('timestamp')))
            db.executemany('INSERT INTO sentences VALUES (?,?,?,?,?,?,?,?,?,?)',pending);db.commit()
        db.execute("INSERT INTO sentence_fts(sentence_fts) VALUES('rebuild')")
        metadata=signature|dict(source_sha256=sha256(source),reviews=reviews,sentences=sentences,sqlite_version=sqlite3.sqlite_version,
            tokenizer='unicode61',bm25=dict(k1=1.2,b=.75,score_sign='API negates SQLite score'),sentence_policy='punctuation/newline; title and body separate; HTML stripped/unescaped',build_s=time.monotonic()-started)
        db.executemany('INSERT INTO metadata VALUES (?,?)',[(k,json.dumps(v)) for k,v in metadata.items()]);db.commit()
        count=db.execute('SELECT count(*) FROM sentences').fetchone()[0]
        if count!=sentences:raise ValueError('Index row count mismatch')
        db.close();db=None;temporary.replace(destination)
        return metadata|dict(index_bytes=destination.stat().st_size,reused=False)
    finally:
        if db is not None:db.close()
        for suffix in ('','-journal'):Path(str(temporary)+suffix).unlink(missing_ok=True)

def probe_index(index):
    """Skip titles containing only emoji/punctuation before testing retrieval."""
    skipped=0
    for row in index.db.execute('SELECT text FROM sentences ORDER BY sentence_id'):
        if not any(c.isalnum() for c in row[0]):skipped+=1;continue
        return dict(has_searchable_sentence=True,retrieval_nonempty=bool(index.retrieve(row[0],1)),skipped_nonword_sentences=skipped)
    return dict(has_searchable_sentence=False,retrieval_nonempty=False,skipped_nonword_sentences=skipped)
