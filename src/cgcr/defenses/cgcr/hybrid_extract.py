"""extractor v2: deterministic typed SPEC mentions plus short non-SPEC LLM lines.

Catalog matching proposes an attribute; it does not license a claim. Unmatched
measurements are retained, and ambiguous attribute assignments stay unmapped.
No attack-specific template or effectiveness signal enters extraction.
"""
from dataclasses import asdict
import re
import time

from cgcr.data.evidence import classify_detail, evidence_details
from cgcr.data.units import NUMBER, COUNT_MENTIONS, parse_attribute, parse_quantity, parse_temperature
from cgcr.defenses.cgcr.catalog import check_catalog
from cgcr.defenses.cgcr.extract import Claim, text_sha
from cgcr.utils.io import write_jsonl

NON_SPEC_TYPES=('USE_CASE','COMPARATIVE','AUTHORITY','TEMPORAL','SUBJECTIVE')
TYPE_PRIORITY=('SPEC','AUTHORITY','COMPARATIVE','TEMPORAL','USE_CASE','SUBJECTIVE')
EXCLUDED_ATTRIBUTES={'package information','model','model number','item model number',
    'part number','manufacturer part number','oem part number'}


def normalized_span(value):
    return ' '.join(str(value).casefold().split())


INCOMPLETE_ENDINGS={'a','an','the','and','or','but','of','for','with','by','to','into','from','than','its','their','our','your'}


def complete_phrase(span):
    """Conservative lexical check; this is not a semantic-accuracy guarantee."""
    words=re.findall(r"[\w'-]+",span.casefold())
    return bool(words) and words[-1] not in INCOMPLETE_ENDINGS and not span.rstrip().endswith('-')


def deduplicate_spans(claims,audit=None,stage='non_spec'):
    """One claim per normalized source span; log conflicting types privately.

    Typed deterministic SPEC wins a cross-stage collision. LLM types use the
    predeclared authority/comparison/time/use-case/subjectivity priority only.
    Within a type, the first claim in deterministic extraction order wins.
    """
    unique={};removed=0;conflicts=[]
    for claim in claims:
        key=normalized_span(claim['span'])
        if key not in unique:unique[key]=claim;continue
        previous=unique[key];removed+=1
        if previous['claim_type']!=claim['claim_type']:
            winner=min((previous,claim),key=lambda c:TYPE_PRIORITY.index(c['claim_type']))
            conflicts.append(dict(span=previous['span'],types=[previous['claim_type'],claim['claim_type']],
                kept_type=winner['claim_type'],stage=stage))
            unique[key]=winner
    if audit is not None:
        audit.setdefault('duplicates_removed',{})[stage]=removed
        audit.setdefault('type_conflicts',[]).extend(conflicts)
    return list(unique.values())
SCALAR_UNITS=(r'fluid\s+ounces?|fl\.?\s*oz\.?|millilit(?:er|re)s?|lit(?:er|re)s?|'
    r'kilograms?|milligrams?|grams?|pounds?|ounces?|lbs?|kg|mg|oz|g|'
    r'gallons?|quarts?|pints?|cups?|ml|l|'
    r'centimet(?:er|re)s?|millimeters?|meters?|inches|inch|feet|foot|cm|mm|in|ft|m|["\']|'
    r'kilowatts?|watts?|kw|w|volts?|v|counts?|pieces?|packs?|'
    r'(?:degrees?\s*|°\s*)?(?:celsius|fahrenheit|kelvin|[cfk])')
SCALAR=re.compile(rf'(?<![\w.]){NUMBER}(?:\s*(?:-|–|—|to)\s*{NUMBER})?\s*-?\s*(?:{SCALAR_UNITS})(?!\w)',re.I)
DUNIT=r'(?:inches|inch|feet|foot|cm|mm|in|ft|m|["\'])'
DIMENSIONS=re.compile(rf'(?<![\w.]){NUMBER}\s*(?:{DUNIT})?\s*[DWLH]?(?:\s*[x×]\s*{NUMBER}\s*(?:{DUNIT})?\s*[DWLH]?){{0,1}}\s*[x×]\s*{NUMBER}\s*{DUNIT}\s*[DWLH]?(?!\w)',re.I)
QUALIFIER=re.compile(r'(?:(?:not|without|never|no|up to|at least|at most|less than|more than|over|under)\s+)$',re.I)


def string_match(phrase,text):
    """Return the actual source bytes; allow case/whitespace changes, never fuzzy matches."""
    phrase=phrase.strip().strip('"“”')
    if not phrase:return None
    pattern=r'(?<!\w)'+r'\s+'.join(re.escape(part) for part in phrase.split())+r'(?!\w)'
    match=re.search(pattern,text,re.I)
    return match.group() if match else None


def _literal(value,text):
    pattern=r'(?<!\w)'+r'\s+'.join(re.escape(p) for p in value.split())+r'(?!\w)'
    return re.finditer(pattern,text,re.I)


def _value(span):
    # A lexical hyphen between number and unit is not a negative sign or numeric range.
    return re.sub(r'(?<=\d)\s*-\s*(?=[A-Za-z°])',' ',span).strip()


def _qualified(text,start,end):
    prefix=text[max(0,start-25):start]
    qualifier=QUALIFIER.search(prefix)
    return text[max(0,start-25)+qualifier.start():end] if qualifier else text[start:end]


def _claim(attribute,value,span,unmapped=False):
    return Claim(attribute=attribute,value=value,unit=None,claim_type='SPEC',span=span).model_dump()|dict(unmapped=unmapped)


def _context_score(entry,text,start,end):
    context=text[max(0,start-60):min(len(text),end+60)]
    ignored={'item','product','of','the','a','an','number'}
    aliases=[entry['canonical_name'],*entry.get('aliases',[])]
    tokens={p.casefold() for alias in aliases for p in re.findall(r'[A-Za-z]+',alias) if p.casefold() not in ignored}
    return sum(bool(re.search(r'\b'+re.escape(token)+r'\b',context,re.I)) for token in tokens)


def deterministic_specs(text,catalog,schema,audit=None):
    """Extract typed value mentions and preserve nonmatching numeric/unit assertions."""
    entries={a['canonical_name']:a for a in schema['attributes'] if classify_detail(a['canonical_name'],schema)=='EVIDENCE'
        and normalized_span(a['canonical_name']) not in EXCLUDED_ATTRIBUTES
        and not re.search(r'\b(?:model|part) (?:number|no\.?|#)\b',normalized_span(a['canonical_name']))}
    parents={k:v for k,v in evidence_details(catalog,schema).items() if k in entries};claims=[]
    if 'manufacturer' in parents:
        brand={normalized_span(v) for v in parents.get('brand',[])}
        parents['manufacturer']=[v for v in parents['manufacturer'] if normalized_span(v) not in brand]
    # Literal categorical properties cannot be licensed merely because they exist in metadata.
    for name,values in parents.items():
        entry=entries[name]
        if entry['value_type']=='numeric+unit':continue
        for value in values:
            if len(value.strip())<3:continue
            for match in _literal(value,text):
                if entry['value_type']=='boolean' and not _context_score(entry,text,match.start(),match.end()):continue
                span=_qualified(text,match.start(),match.end())
                claims.append(_claim(name,span,span))
    # Largest dimension expressions win over their constituent scalar measurements.
    measurements=list(DIMENSIONS.finditer(text))+list(COUNT_MENTIONS.finditer(text))
    measurements += [m for m in SCALAR.finditer(text) if not any(m.start()<d.end() and m.end()>d.start() for d in measurements)]
    for mention in sorted(measurements,key=lambda m:(m.start(),-len(m.group()))):
        raw=mention.group().strip();value=_value(raw)
        if len(value)<3:continue
        compatible=[a for a in entries.values() if a['value_type']=='numeric+unit' and parse_attribute(value,a['canonical_name']) is not None]
        matched=[a for a in compatible if check_catalog(a['canonical_name'],value,catalog,schema,rel_tolerance=0).status=='support']
        count=parse_quantity(value,'count')
        # Explicit lexical aliases disambiguate the three distinct schema counts;
        # a parent value matching another count is not a reason to relabel it.
        preferred=('number of pieces' if re.search(r'\b(?:pieces?|pcs?)\b',raw,re.I) else
            'number of items' if re.search(r'\b(?:set|packs?|items?)\b',raw,re.I) else 'unit count') if count else None
        aliases=[a for a in compatible if a['canonical_name']==preferred or
            any(normalized_span(alias)==preferred for alias in a.get('aliases',[]))]
        choices=aliases or matched or [a for a in compatible if a['canonical_name'] in parents] or compatible
        scores=[(_context_score(a,text,mention.start(),mention.end()),a) for a in choices]
        chosen=None
        if len(choices)==1:chosen=choices[0]
        elif scores:
            best=max(s for s,a in scores);winners=[a for s,a in scores if s==best]
            if best>0 and len(winners)==1:chosen=winners[0]
        if chosen:name=chosen['canonical_name']
        else:
            dimension=next((d for d in ('weight','volume','dimensions','wattage','voltage','count') if parse_quantity(value,d)),None)
            if dimension is None and parse_temperature(value):dimension='temperature'
            name='unmapped '+(dimension or 'measurement')
        span=_qualified(text,mention.start(),mention.end())
        claims.append(_claim(name,_value(span),span,chosen is None))
    # Numeric fields unsupported by the unit helper still get exact literal extraction.
    for name,values in parents.items():
        if entries[name]['value_type']!='numeric+unit':continue
        for value in values:
            if len(value.strip())<3:continue
            if parse_attribute(value,name) is not None:continue
            for match in _literal(value,text):
                span=_qualified(text,match.start(),match.end())
                claims.append(_claim(name,span,span))
    claims=deduplicate_spans(claims,audit,'spec');unique={};removed=0
    for claim in claims:
        parsed=parse_attribute(claim['value'],claim['attribute'])
        value=(parsed.dimension,parsed.unit,parsed.values,parsed.axes) if parsed else normalized_span(claim['value'])
        key=(normalized_span(claim['attribute']),value)
        if key in unique:removed+=1
        else:unique[key]=claim
    if audit is not None:audit.setdefault('duplicates_removed',{})['spec_values']=removed
    return sorted(unique.values(),key=lambda c:(text.find(c['span']),c['attribute'],c['value']))


def parse_non_spec(raw,text,completion_tokens=None,audit=None):
    claims=[];errors=[];lines=[s.strip() for s in raw.splitlines() if s.strip()]
    if not lines:errors.append('empty_output')
    if len(lines)==1 and lines[0].upper()=='NONE':return [],(['token_limit'] if completion_tokens and completion_tokens>=256 else [])
    if len(lines)>15:errors.append('more_than_15_lines')
    if completion_tokens and completion_tokens>=256:errors.append('token_limit')
    for index,line in enumerate(lines[:15]):
        if '|' not in line:errors.append(f'line_{index}:format');continue
        kind,phrase=line.split('|',1);kind=kind.strip().upper();phrase=phrase.strip()
        if kind not in NON_SPEC_TYPES:errors.append(f'line_{index}:type');continue
        span=string_match(phrase,text)
        if span is None:errors.append(f'line_{index}:not_in_source');continue
        if not complete_phrase(span):errors.append(f'line_{index}:truncated_phrase');continue
        if len(span)>350:errors.append(f'line_{index}:phrase_too_long');continue
        claim=Claim(attribute=kind.lower().replace('_',' '),value=span,unit=None,claim_type=kind,span=span).model_dump()|dict(unmapped=False)
        claims.append(claim)
    return deduplicate_spans(claims,audit,'non_spec'),errors


class HybridExtractor:
    def __init__(self,backend,schema,prompt,version):
        self.backend,self.schema,self.prompt,self.version=backend,schema,prompt,version

    def extract(self,item_id,text,catalog,seed):
        started=time.perf_counter();audit={};specs=deterministic_specs(text,catalog,self.schema,audit)
        deterministic_seconds=time.perf_counter()-started
        output=self.backend.generate([dict(role='system',content=self.prompt),dict(role='user',content=text)],seed=seed,max_tokens=256,temperature=0.)
        non_spec,errors=parse_non_spec(output.text,text,output.completion_tokens,audit)
        claims=deduplicate_spans(specs+non_spec,audit,'cross_stage')
        attempt=asdict(output);attempt['status']='invalid' if errors else 'accepted'
        attempt['reason_code']='line_validation' if errors else 'accepted'
        return dict(extractor_family='v2-hybrid',extractor_version=self.version,item_id=item_id,text_sha256=text_sha(text),
            seed=seed,status='partial_failure' if errors else 'ok',claims=claims,postprocess=audit,
            rejected_claims=[dict(reason_code=e) for e in errors],attempts=[attempt],
            deterministic_seconds=deterministic_seconds,elapsed_s=time.perf_counter()-started,
            deterministic_spec_claims=sum(c['claim_type']=='SPEC' for c in claims),
            unmapped_spec_claims=sum(c['claim_type']=='SPEC' and c.get('unmapped',False) for c in claims),
            non_spec_claims=sum(c['claim_type']!='SPEC' for c in claims))

    def cached(self,path,item_id,text,catalog,seed):
        from pathlib import Path
        import json
        path=Path(path)
        if path.exists():
            value=json.loads(path.read_text())
            if (value['item_id'],value['text_sha256'],value['extractor_version'])!=(item_id,text_sha(text),self.version):raise ValueError('Hybrid cache identity mismatch')
            return value,True
        value=self.extract(item_id,text,catalog,seed);path.parent.mkdir(parents=True,exist_ok=True);write_jsonl(path,[value])
        return value,False
