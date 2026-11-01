"""cached-score thresholding and constrained deterministic selection."""
from copy import deepcopy
from cgcr.defenses.cgcr.calibration import relicense, choose_setting


def test_relicense_uses_cached_scores_without_mutating_reference():
    d=dict(claim=dict(claim_type='USE_CASE',value='storing rice'),trace={'reviews':{'trace':[
        dict(entailment=.8,contradiction=.1,neutral=.1,user_id='u',verified_purchase=True)]}},licensed=False,v=0.)
    original=deepcopy(d)
    assert relicense(d,.6,'mean')['licensed']
    assert not relicense(d,.8,'reviewer_capped')['licensed']
    assert d==original


def test_selection_respects_all_inserted_constraint_and_no_feasible_case():
    def cell(tau,clean,inserted):
        return dict(tau=tau,aggregator='mean',clean_use_case={'share':clean},inserted_all={'share':inserted})
    unsafe=cell(.4,.9,.051);safe=cell(.6,.4,.05)
    assert choose_setting([unsafe,safe])==safe
    assert choose_setting([unsafe]) is None
    assert choose_setting([cell(.5,.4,.05),safe])==safe


def test_clarified_use_case_scope_preserves_all_inserted_reporting():
    cells=[dict(tau=.4,aggregator='trimmed_mean',clean_use_case={'share':.1},
                inserted_use_case={'share':.03},inserted_all={'share':.09})]
    original=deepcopy(cells)
    assert choose_setting(cells) is None
    assert choose_setting(cells,constraint_scope='use_case')==cells[0]
    assert cells==original
