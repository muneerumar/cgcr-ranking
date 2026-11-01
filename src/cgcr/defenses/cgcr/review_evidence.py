"""CPU skeleton: catalog and review-evidence components; no  extraction.

No full verifier run, calibrated fusion or licensing decision is implemented.
AUTHORITY/TEMPORAL/COMPARATIVE require external support; SUBJECTIVE is unlicensed.
Callers supply claims and choose an explicit robust review estimator.
"""
from dataclasses import asdict
from cgcr.defenses.cgcr.catalog import check_catalog
from cgcr.defenses.cgcr.aggregation import aggregate_reviews

TYPE_PRIORS={'AUTHORITY':'requires_external','TEMPORAL':'requires_external','COMPARATIVE':'requires_external','SUBJECTIVE':'never'}

def review_evidence(item_id,claim,retriever,nli,*,claim_type='SPEC',k=20,estimator='reviewer_capped',**aggregation):
    prior=TYPE_PRIORS.get(claim_type)
    if prior:return dict(status=prior,score=0.,trace=[])
    if claim_type not in ('SPEC','USE_CASE'):return dict(status='unknown_claim_type',score=0.,trace=[])
    sentences=retriever.retrieve(item_id,claim,k)
    scores=nli.score([(s.text,claim) for s in sentences]) if sentences else []
    if len(scores)!=len(sentences):raise ValueError('NLI result count mismatch')
    rows=[asdict(sentence)|asdict(score) for sentence,score in zip(sentences,scores)]
    result=aggregate_reviews(rows,strategy=estimator,**aggregation)
    return dict(status='scored' if sentences else 'unknown_no_reviews',score=result.support_strength,aggregate=asdict(result),trace=rows,
                model_id=getattr(nli,'model_id',None),revision=getattr(nli,'revision',None))
