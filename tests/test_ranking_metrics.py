"""hand-computed linear-gain NDCG, eligibility, Exact recall and tau."""
import math
import pytest
from cgcr.metrics.ranking import ndcg_at_k,recall_exact_at_k,kendall_tau


def test_hand_computed_ndcg_and_gains():
    expected=(.1+1/math.log2(3))/(1+.1/math.log2(3))
    assert ndcg_at_k(['E','S'],[1,0],2,utility_eligible=True)==pytest.approx(expected)
    assert ndcg_at_k(['E','S','I'],[0,1,2],2,utility_eligible=True)==1
    assert ndcg_at_k(['E','S'],[1,0],1,{'E':2,'S':1},utility_eligible=True)==.5
    assert ndcg_at_k(['E','E'],[1,0],utility_eligible=False) is None
    assert ndcg_at_k(['I','I'],[0,1],utility_eligible=True)==0


def test_recall_and_kendall():
    assert recall_exact_at_k(['E','E','S'],[0,2,1],2)==.5
    assert recall_exact_at_k(['S'],[0]) is None
    assert kendall_tau([0,1,2],[0,1,2])==1
    assert kendall_tau([0,1,2],[2,1,0])==-1
    assert kendall_tau([0,1,2],[0,2,1])==pytest.approx(1/3)
    with pytest.raises(ValueError):ndcg_at_k(['E','S'],[0,0],utility_eligible=True)
