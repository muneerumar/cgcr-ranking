"""immutable  cohort/artifact selection and private content inventory."""
import json
import random
from pathlib import Path
from cgcr.defenses.cgcr.extract import sha,text_sha,cache_key
from cgcr.rankers.budget import TextBudget,TextEdit
from cgcr.utils.io import write_jsonl

def load(path):return json.loads(Path(path).read_text())
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);write_jsonl(path,[value])
def file_sha(path):
    import hashlib
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk:=stream.read(1024*1024):h.update(chunk)
    return h.hexdigest()
def inserted_intervals(edits):
    """Translate clean-coordinate non-overlapping edits to attacked coordinates."""
    shift=0;out=[]
    for e in sorted(edits,key=lambda e:(e['start'],e['end'])):
        start=e['start']+shift;end=start+len(e['replacement'])
        if end>start:out.append((start,end))
        shift+=len(e['replacement'])-(e['end']-e['start'])
    return out

def claim_origin(span,text,intervals):
    starts=[];start=0
    while (start:=text.find(span,start))>=0:starts.append(start);start+=1
    if not starts:return 'invalid_span'
    states=[]
    for start in starts:
        end=start+len(span)
        if any(start<b and end>a for a,b in intervals):states.append('inserted')
        else:states.append('clean')
    return states[0] if len(set(states))==1 else 'ambiguous'


