"""automatic fragment diagnostics; proxies, never human accuracy labels."""
from collections import Counter
import re
from cgcr.defenses.cgcr.hybrid_extract import complete_phrase

CHECKABLE=frozenset({'SPEC','USE_CASE'})
UNIT_WORDS=frozenset('x d w h l oz fl lb lbs pound pounds ounce ounces g kg gram grams ml liter liters litre litres inch inches in cm mm ft feet foot count counts pack packs pieces piece pcs set of size capacity weight dimensions item product'.split())


def words(text):return re.findall(r"[A-Za-z]+(?:['’-][A-Za-z]+)*|\d+(?:\.\d+)?",str(text))
def coarse(kind):return 'CHECKABLE' if kind in CHECKABLE else 'NON_LICENSABLE'


def malformed_numeric(value):
    """Reject broken/zero/negative dimension axes; ordinary ranges remain unknown."""
    text=str(value).strip()
    if not re.search(r'\d',text):return False
    if re.search(r'''\d[\d.\s"'DWHL]*[x×]\s*(?:-?\d|$)''',text,re.I):
        axes=re.split(r'\s*[x×]\s*',text,flags=re.I)
        for axis in axes:
            match=re.search(r'(?<!\w)-?\d+(?:\.\d+)?',axis)
            if match is None or float(match.group())<=0:return True
        if re.search(r'[x×]\s*$',text,re.I):return True
    return bool(re.search(r'\d\s*\.\s*[x×]|\d\s*[x×]\s*[^\w\d]*$',text,re.I))


def fragment_flags(claim,text=None):
    span=claim.get('span','');tokens=words(span);alpha=[w.casefold() for w in tokens if re.search('[A-Za-z]',w)]
    flags=[]
    if len(tokens)<2:flags.append('fewer_than_two_words')
    if re.search(r'\d',span) and not [w for w in alpha if w not in UNIT_WORDS]:flags.append('mostly_digits_units')
    if not complete_phrase(span):flags.append('truncated')
    if text is not None:
        starts=[m.start() for m in re.finditer(re.escape(span),text)] if span else []
        if not starts or not any((s==0 or not(text[s-1].isalnum() and span[0].isalnum())) and
            (s+len(span)==len(text) or not(text[s+len(span)].isalnum() and span[-1].isalnum())) for s in starts):flags.append('truncated_or_not_verbatim')
    if malformed_numeric(claim.get('value',span)):flags.append('malformed_numeric')
    return flags


def quality_summary(outputs):
    counts={};classes=Counter();texts=0
    for output in outputs:
        texts+=1
        for c in output['claims']:
            kind=c['claim_type'];row=counts.setdefault(kind,Counter());row['claims']+=1;classes[coarse(kind)]+=1
            flags=fragment_flags(c,output.get('source_text'))
            row['fragmented_claims']+=bool(flags);row.update(flags)
    result={kind:dict(row,fragment_rate=row['fragmented_claims']/row['claims']) for kind,row in counts.items()}
    n=sum(r['claims'] for r in counts.values());bad=sum(r['fragmented_claims'] for r in counts.values())
    ns=sum(r['claims'] for k,r in counts.items() if k!='SPEC');nb=sum(r['fragmented_claims'] for k,r in counts.items() if k!='SPEC')
    return dict(texts=texts,claims=n,fragmented_claims=bad,fragment_rate=bad/n if n else None,
        non_spec_claims=ns,non_spec_fragment_rate=nb/ns if ns else 0.,per_type=result,
        coarse_counts=dict(classes),coarse_shares={k:v/n for k,v in classes.items()} if n else {},
        caveat='Heuristic flags, not semantic accuracy. One-word categorical SPEC values can be valid; report by type.')
