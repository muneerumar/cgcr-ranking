"""calibrated verification and masked-catalog SPEC reconstruction."""
from copy import deepcopy
from cgcr.defenses.cgcr.verify import verify_claim
from cgcr.defenses.cgcr.review_evidence import review_evidence
from cgcr.defenses.cgcr.hybrid_extract import deterministic_specs,deduplicate_spans
from cgcr.defenses.cgcr.claim_quality import malformed_numeric


def verify_final_claim(claim,row,schema,config,*,retriever=None,nli=None):
    if claim['claim_type']!='USE_CASE':return verify_claim(claim,row,schema,config,retriever=retriever,nli=nli)
    evidence=review_evidence(row['item_id'],claim['value'].strip(),retriever,nli,claim_type='USE_CASE',
        k=config.get('top_k',20),estimator=config['aggregator'],per_reviewer_cap=1.,verified_weight=1.,unverified_weight=.3,helpful_log_bonus=.1)
    licensed=evidence['status']=='scored' and evidence['score']>=config['license_threshold']
    return dict(claim=claim,licensed=licensed,v=evidence['score'],status='review_support' if licensed else evidence['status'],
        trace={'reviews':evidence},render_attribute='use case',render_value=' '.join(claim['value'].split()) if licensed else None)


def masked_claims(extraction,text,row,schema,family):
    """The deterministic extractor sees only kept details; reuse text-only LLM claims."""
    specs=deterministic_specs(text,row['catalog'],schema)
    if family=='v3-hybrid':specs=[c for c in specs if not malformed_numeric(c['value'])]
    non_spec=[deepcopy(c) for c in extraction['claims'] if c['claim_type']!='SPEC']
    return deduplicate_spans(specs+non_spec)
