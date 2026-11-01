"""schema-constrained, span-validated atomic claims and content-addressed cache."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from cgcr.defenses.cgcr.catalog import canonical_attribute
from cgcr.utils.io import write_jsonl

CLAIM_TYPES=('SPEC','USE_CASE','COMPARATIVE','AUTHORITY','TEMPORAL','SUBJECTIVE')
class Claim(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    attribute:str=Field(min_length=1,max_length=120)
    value:str=Field(min_length=1,max_length=350)
    unit:str|None=Field(max_length=60)
    claim_type:Literal['SPEC','USE_CASE','COMPARATIVE','AUTHORITY','TEMPORAL','SUBJECTIVE']
    span:str=Field(min_length=1,max_length=700)
class ClaimEnvelope(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True)
    claims:list[Claim]=Field(max_length=32)

def sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def text_sha(text):return hashlib.sha256(text.encode()).hexdigest()
def cache_key(item_id,text,version):return sha([item_id,text_sha(text),version])
def validate_claims(raw,text,schema):
    envelope=ClaimEnvelope.model_validate_json(raw)
    out=[]
    for claim in envelope.claims:
        if claim.span not in text:raise ValueError('span_not_verbatim')
        row=claim.model_dump();mapped=canonical_attribute(claim.attribute,schema)
        row['attribute']=mapped or claim.attribute.strip().casefold()
        row['unmapped']=claim.claim_type=='SPEC' and mapped is None
        out.append(row)
    return out

class Extractor:
    def __init__(self,backend,schema,prompt,repair_prompt,version,max_tokens=2048):
        self.backend,self.schema,self.prompt,self.repair_prompt=backend,schema,prompt,repair_prompt
        self.version,self.max_tokens=version,max_tokens
    def extract(self,item_id,text,seed):
        names=[a['canonical_name'] for a in self.schema['attributes']]
        messages=[{'role':'system','content':self.prompt.replace('{attributes}',json.dumps(names))},
                  {'role':'user','content':text}]
        attempts=[];partial=None
        for attempt in range(2):
            output=self.backend.generate_json(messages,json_schema=ClaimEnvelope.model_json_schema(),seed=seed,max_tokens=self.max_tokens)
            record=asdict(output)
            try:
                envelope=ClaimEnvelope.model_validate_json(output.text)
                valid=[];rejected=[]
                for index,claim in enumerate(envelope.claims):
                    if claim.span not in text:rejected.append(dict(index=index,claim=claim.model_dump(),reason_code='span_not_verbatim'))
                    else:valid+=validate_claims(json.dumps({'claims':[claim.model_dump()]}),text,self.schema)
                if rejected:
                    partial=dict(claims=valid,rejected_claims=rejected,source_attempt=attempt)
                    raise ValueError('span_not_verbatim')
                claims=valid
                record['status']='accepted';attempts.append(record)
                return dict(status='ok' if attempt==0 else 'repaired',item_id=item_id,text_sha256=text_sha(text),
                    extractor_version=self.version,claims=claims,rejected_claims=[],attempts=attempts,seed=seed)
            except (ValidationError,ValueError) as exc:
                # Never place raw output or validation details in public logs.
                record.update(status='invalid',reason_code='schema_validation' if isinstance(exc,ValidationError) else 'span_not_verbatim')
                attempts.append(record)
                if attempt==0:
                    errors=[dict(index=r['index'],reason=r['reason_code']) for r in partial['rejected_claims']] if partial else [{'reason':'schema_validation'}]
                    repair=self.repair_prompt.replace('{errors}',json.dumps(errors))
                    messages += [{'role':'\u0061ssistant','content':output.text},{'role':'user','content':repair}]
        if partial and partial['claims']:
            return dict(status='partial_failure',item_id=item_id,text_sha256=text_sha(text),extractor_version=self.version,
                claims=partial['claims'],rejected_claims=partial['rejected_claims'],attempts=attempts,seed=seed,
                accepted_from_attempt=partial['source_attempt'],failure_policy='valid_spans_only; invalid claims logged and unlicensed')
        return dict(status='extraction_failed',item_id=item_id,text_sha256=text_sha(text),extractor_version=self.version,
                    claims=[],rejected_claims=partial['rejected_claims'] if partial else [],attempts=attempts,seed=seed,failure_policy='metadata_only_rendering_with_explicit_failure_flag')
    def cached(self,path,item_id,text,seed):
        path=Path(path)
        if path.exists():
            value=json.loads(path.read_text())
            if (value['item_id'],value['text_sha256'],value['extractor_version'])!=(item_id,text_sha(text),self.version):raise ValueError('Extraction cache identity mismatch')
            return value,True
        value=self.extract(item_id,text,seed);path.parent.mkdir(parents=True,exist_ok=True)
        write_jsonl(path,[value]);return value,False
