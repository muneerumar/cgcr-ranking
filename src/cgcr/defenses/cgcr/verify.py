"""fail-closed claim licensing with catalog checks and CPU review NLI traces."""
from dataclasses import asdict
from cgcr.data.evidence import evidence_details,classify_detail
from cgcr.data.units import parse_attribute
from cgcr.defenses.cgcr.catalog import canonical_attribute,check_catalog
from cgcr.defenses.cgcr.review_evidence import review_evidence

EXTERNAL_TYPES=frozenset({'AUTHORITY','TEMPORAL','COMPARATIVE'})
def claim_hypothesis(claim):
    # Score the extracted assertion, not an embedded endorsement or source instruction.
    if claim.get('extractor_family')=='v2-hybrid':return claim['value'].strip()
    if claim['claim_type']=='USE_CASE':return 'This product is suitable for '+claim['value'].strip().rstrip('.')+'.'
    unit=claim.get('unit');value=claim['value']
    if unit and parse_attribute(value,claim['attribute']) is None:value=value+' '+unit
    return claim['attribute']+': '+value

def verify_claim(claim,row,schema,config,*,retriever=None,nli=None):
    threshold=config['license_threshold']
    if not 0<=threshold<=1:raise ValueError('License threshold must be in [0,1]')
    result=dict(claim=claim,licensed=False,v=0.,status='unknown',trace={},render_attribute=None,render_value=None)
    kind=claim['claim_type']
    if kind in EXTERNAL_TYPES:return result|dict(status='unlicensed_external_support_unavailable')
    if kind=='SUBJECTIVE':return result|dict(status='unlicensed_subjective')
    if kind=='SPEC':
        name=canonical_attribute(claim['attribute'],schema)
        if name is None or claim.get('unmapped'):return result|dict(status='unmapped')
        if classify_detail(name,schema)!='EVIDENCE':return result|dict(status='excluded_attackable')
        entry=next(a for a in schema['attributes'] if a['canonical_name']==name)
        value=claim['value'];unit=claim.get('unit')
        if unit and entry['value_type']=='numeric+unit' and parse_attribute(value,name) is None:value=value+' '+unit
        variant=row.get('joined_via_variant',row.get('product_id')!=row['item_id'])
        # Sensitive child claims require an exact normalized parent match, not 5% tolerance.
        tolerance=0. if variant and entry['variant_sensitive'] else config.get('numeric_rel_tolerance',.05)
        decision=check_catalog(name,value,row['catalog'],schema,joined_via_variant=variant,rel_tolerance=tolerance)
        result.update(status=decision.status,trace={'catalog':asdict(decision)},render_attribute=name)
        if decision.status=='support':
            matching=[]
            for parent in decision.parent_values:
                isolated=dict(row['catalog'],details={name:parent})
                d=check_catalog(name,value,isolated,schema,joined_via_variant=variant,rel_tolerance=tolerance)
                if d.status=='support':matching.append(parent)
            if not matching:raise ValueError('Supported catalog claim has no matching platform value')
            # Render the platform value, never an attacker value within numeric tolerance.
            result.update(v=1.,licensed=True,render_value=sorted(matching)[0])
        return result
    if kind=='USE_CASE':
        if retriever is None or nli is None:raise ValueError('USE_CASE verification requires review retrieval and CPU NLI')
        evidence=review_evidence(row['item_id'],claim_hypothesis(claim),retriever,nli,claim_type=kind,
            k=config.get('top_k',20),estimator='reviewer_capped',per_reviewer_cap=1.,verified_weight=1.,unverified_weight=.3,helpful_log_bonus=.1)
        licensed=evidence['status']=='scored' and evidence['score']>=threshold
        return result|dict(status='review_support' if licensed else evidence['status'],v=evidence['score'],licensed=licensed,
            trace={'reviews':evidence},render_attribute='use case',render_value=' '.join(claim['value'].split()) if licensed else None)
    return result|dict(status='unknown_claim_type')
