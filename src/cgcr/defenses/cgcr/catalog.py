"""CPU skeleton: typed structured-detail checks with normalized units.

Seller titles/features/text never enter this check. Numeric tolerance is relative
to the platform value. Parent mismatches on variant-sensitive keys remain unknown.
"""
from dataclasses import dataclass
import math
from cgcr.data.evidence import classify_detail,evidence_details
from cgcr.data.schema_builder import normalize_key
from cgcr.data.units import parse_attribute,unit_parse_status

@dataclass(frozen=True)
class CatalogDecision:
    status:str
    attribute:str|None
    parent_values:tuple[str,...]
    reason:str

def canonical_attribute(attribute,schema):
    normalized=normalize_key(attribute)
    for entry in schema.get('attributes',[]):
        if normalized in {normalize_key(v) for v in [entry['canonical_name'],*entry['aliases']]}:return entry['canonical_name']
    return None

def _boolean(value):
    text=str(value).strip().casefold()
    if text in ('yes','true','1'):return True
    if text in ('no','false','0'):return False
    return None

def _aligned(claim,parent):
    if claim.unit!=parent.unit or claim.dimension!=parent.dimension or len(claim.values)!=len(parent.values):return None
    if claim.axes and parent.axes:
        if len(set(claim.axes))!=len(claim.axes) or set(claim.axes)!=set(parent.axes):return None
        cv=dict(zip(claim.axes,claim.values));pv=dict(zip(parent.axes,parent.values))
        return [cv[k] for k in sorted(cv)],[pv[k] for k in sorted(pv)]
    if bool(claim.axes)!=bool(parent.axes):return None
    return claim.values,parent.values

def check_catalog(attribute,value,catalog,schema,*,joined_via_variant=False,rel_tolerance=.05,abs_tolerance=1e-6):
    if not math.isfinite(rel_tolerance) or not math.isfinite(abs_tolerance) or rel_tolerance<0 or abs_tolerance<0:raise ValueError('Tolerances must be nonnegative')
    name=canonical_attribute(attribute,schema)
    if classify_detail(name,schema)!='EVIDENCE':
        return CatalogDecision('excluded_attackable' if name else 'unknown',name,(),'Attribute is absent or not licensed as structured evidence')
    entry=next(a for a in schema['attributes'] if a['canonical_name']==name)
    parents=tuple(evidence_details(catalog,schema).get(name,[]))
    if not parents:return CatalogDecision('unknown',name,parents,'Missing platform value')
    if unit_parse_status(value,name)=='range_unparsed' or any(unit_parse_status(p,name)=='range_unparsed' for p in parents):
        return CatalogDecision('range_unparsed',name,parents,'Range conversion is deliberately unimplemented')
    normalize=lambda x:' '.join(str(x).casefold().split())
    comparable=False;support=False
    if entry['value_type']=='numeric+unit':
        claim=parse_attribute(value,name)
        if claim is not None:
            for raw in parents:
                parent=parse_attribute(raw,name)
                pair=_aligned(claim,parent) if parent else None
                if pair is None:continue
                comparable=True
                support|=all(abs(c-p)<=abs(p)*rel_tolerance+abs_tolerance for c,p in zip(*pair))
    elif entry['value_type']=='boolean':
        claim=_boolean(value);values=[_boolean(p) for p in parents]
        comparable=claim is not None and any(v is not None for v in values)
        support=comparable and any(v is not None and v==claim for v in values)
    else:
        comparable=True;support=any(normalize(value)==normalize(p) for p in parents)
    if support:return CatalogDecision('support',name,parents,'Normalized value matches the platform record within configured tolerance')
    if not comparable:return CatalogDecision('unknown',name,parents,'Unit, axis or boolean conversion unavailable')
    if joined_via_variant and entry['variant_sensitive']:
        return CatalogDecision('unknown_variant',name,parents,'Parent mismatch cannot contradict an unverified child variant')
    return CatalogDecision('contradiction',name,parents,'Comparable platform value mismatches the claim')
