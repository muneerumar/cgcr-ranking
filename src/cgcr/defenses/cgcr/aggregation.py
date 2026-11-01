"""CPU skeleton: weighted review means, trimming and reviewer contribution caps.

These are score components, not calibrated licensing decisions. Missing reviewer
IDs share a review-level fallback key; identity across anonymous reviews is unknown.
"""
from collections import defaultdict
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class ReviewAggregate:
    entailment:float
    contradiction:float
    neutral:float
    support_strength:float
    effective_weight:float
    evidence_count:int
    reviewer_weights:dict
    strategy:str

def aggregate_reviews(rows,strategy='reviewer_capped',*,trim_fraction=.1,per_reviewer_cap=1.,verified_weight=1.,unverified_weight=.3,helpful_log_bonus=.1,time_window_ms=None,per_window_cap=None):
    if strategy not in ('mean','trimmed_mean','reviewer_capped'):raise ValueError('Unknown estimator')
    params=(trim_fraction,per_reviewer_cap,verified_weight,unverified_weight,helpful_log_bonus)
    if any(not math.isfinite(v) for v in params) or not 0<=trim_fraction<.5 or per_reviewer_cap<=0 or min(verified_weight,unverified_weight,helpful_log_bonus)<0:raise ValueError('Invalid aggregation parameters')
    if (time_window_ms is None)!=(per_window_cap is None) or (time_window_ms is not None and (not math.isfinite(time_window_ms) or not math.isfinite(per_window_cap) or time_window_ms<=0 or per_window_cap<=0)):raise ValueError('Both positive time-window parameters are required')
    data=[]
    for number,row in enumerate(rows):
        probabilities=[float(row[k]) for k in ('entailment','contradiction','neutral')]
        if any(not math.isfinite(v) or not 0<=v<=1 for v in probabilities) or not math.isclose(sum(probabilities),1.,abs_tol=1e-5):raise ValueError('Invalid NLI probabilities')
        helpful=max(0,int(row.get('helpful_vote') or 0))
        weight=(verified_weight if row.get('verified_purchase') else unverified_weight)*(1+helpful_log_bonus*math.log1p(helpful))
        reviewer=row.get('user_id') or 'review:'+str(row.get('review_id',number))
        row=dict(row,**dict(zip(('entailment','contradiction','neutral'),probabilities)))
        data.append(dict(row=row,reviewer=reviewer,weight=weight,order=number))
    if strategy=='trimmed_mean':
        data.sort(key=lambda d:(d['row']['entailment']-d['row']['contradiction'],d['order']))
        trim=int(len(data)*trim_fraction)
        if trim:data=data[trim:-trim]
    if strategy=='reviewer_capped':
        totals=defaultdict(float)
        for d in data:totals[d['reviewer']]+=d['weight']
        for d in data:d['weight']*=min(1.,per_reviewer_cap/totals[d['reviewer']]) if totals[d['reviewer']] else 0.
        if time_window_ms is not None:
            windows=defaultdict(float)
            for d in data:
                stamp=d['row'].get('timestamp');d['window']=int(stamp)//time_window_ms if stamp is not None else 'unknown'
                windows[d['window']]+=d['weight']
            for d in data:d['weight']*=min(1.,per_window_cap/windows[d['window']]) if windows[d['window']] else 0.
    total=sum(d['weight'] for d in data);reviewers=defaultdict(float)
    for d in data:reviewers[d['reviewer']]+=d['weight']
    means={key:sum(d['weight']*d['row'][key] for d in data)/total if total else 0. for key in ('entailment','contradiction','neutral')}
    return ReviewAggregate(**means,support_strength=max(0.,min(1.,means['entailment']-means['contradiction'])),effective_weight=total,evidence_count=len(data),reviewer_weights=dict(reviewers),strategy=strategy)
