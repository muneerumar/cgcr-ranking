"""dev-only review-score recalibration from immutable  NLI traces.

No extraction, retrieval, NLI inference or ranking outcome is used in selection.
The constraint scope is explicit. The initial audit covered all inserted claims;
the user's clarified review-licensing rule constrains inserted USE_CASE claims.
Both shares remain in the immutable sweep cells.
"""
from copy import deepcopy

from cgcr.defenses.cgcr.aggregation import aggregate_reviews


def relicense(decision, tau, aggregator):
    if not 0 <= tau <= 1:
        raise ValueError('Invalid license threshold')
    result = deepcopy(decision)
    if result['claim']['claim_type'] != 'USE_CASE':
        return result
    evidence = result['trace']['reviews']
    rows = evidence['trace']
    aggregate = aggregate_reviews(rows, strategy=aggregator)
    licensed = bool(rows) and aggregate.support_strength >= tau
    result.update(v=aggregate.support_strength, licensed=licensed,
                  status='review_support' if licensed else 'scored' if rows else 'unknown_no_reviews')
    from dataclasses import asdict
    evidence.update(score=aggregate.support_strength, aggregate=asdict(aggregate))
    result['render_attribute'] = 'use case' if licensed else None
    result['render_value'] = ' '.join(result['claim']['value'].split()) if licensed else None
    return result


def choose_setting(cells, maximum_inserted_share=.05, constraint_scope='all'):
    """Max clean USE_CASE share; ties: smaller inserted share, larger tau, name.

    No feasible cell returns None; never silently relax the user's constraint.
    Undefined clean/inserted denominators cannot establish feasibility.
    """
    if constraint_scope not in ('all','use_case'):
        raise ValueError('Unknown inserted-claim constraint scope')
    key='inserted_all' if constraint_scope=='all' else 'inserted_use_case'
    valid = [c for c in cells if c['clean_use_case']['share'] is not None
             and c[key]['share'] is not None
             and c[key]['share'] <= maximum_inserted_share]
    if not valid:
        return None
    return sorted(valid, key=lambda c: (-c['clean_use_case']['share'],
                  c[key]['share'], -c['tau'], c['aggregator']))[0]
