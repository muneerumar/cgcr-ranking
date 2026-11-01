"""project NRG, unconditional promotion/reach and paired query bootstraps.

NRG follows the FramingBench headroom normalization, not GEO-Bench's
list-length denominator. None denotes undefined NRG for a clean rank of one.
"""
import numpy as np


def case_metrics(clean, attacked, n=10):
    if not 1<=clean<=n or not 1<=attacked<=n:
        raise ValueError('Ranks must be one-based and within the complete list')
    return {'NRG':(clean-attacked)/(clean-1) if clean>1 else None,
            **{f'Promote@{k}':float(clean>k and attacked<=k) for k in (1,3,5)},
            'Success@1':float(attacked==1)}


def bootstrap_mean(values, seed=13, replicates=10000):
    values=np.array([v for v in values if v is not None],dtype=float)
    if not len(values):return {'mean':None,'ci95':[None,None],'n':0}
    rng=np.random.default_rng(seed)
    # Bound temporary memory even for the future full-study population.
    draws=[]
    for start in range(0,replicates,250):
        samples=rng.integers(0,len(values),(min(250,replicates-start),len(values)))
        draws.extend(values[samples].mean(axis=1).tolist())
    return {'mean':float(values.mean()),'ci95':np.quantile(draws,[.025,.975]).tolist(),'n':len(values)}


def summarize_cases(cases, seed=13, replicates=10000):
    metrics=[case_metrics(r['clean_rank'],r['attacked_rank']) for r in cases]
    return {key:bootstrap_mean([m[key] for m in metrics],seed,replicates)
            for key in ('NRG','Promote@1','Promote@3','Promote@5','Success@1')}


def paired_position_difference(left, right, seed=13, replicates=10000):
    """Bootstrap paired per-query differences, excluding undefined pairs only."""
    a={r['query_id']:case_metrics(r['clean_rank'],r['attacked_rank']) for r in left}
    b={r['query_id']:case_metrics(r['clean_rank'],r['attacked_rank']) for r in right}
    if a.keys()!=b.keys():raise ValueError('Position comparisons must be paired')
    return {key:bootstrap_mean([a[q][key]-b[q][key] if a[q][key] is not None and b[q][key] is not None else None
                               for q in sorted(a)],seed,replicates)
            for key in ('NRG','Promote@1','Promote@3','Promote@5','Success@1')}
