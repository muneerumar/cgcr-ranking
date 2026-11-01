"""eligibility-filtered linear-gain NDCG, Exact recall, Kendall tau-b."""
import math

DEFAULT_GAINS = {'E': 1., 'S': .1, 'C': .01, 'I': 0.}


def ndcg_at_k(labels, permutation, k=10, gains=None, *, utility_eligible):
    """Return None on excluded sets. Gains are direct relevance, not 2**gain-1."""
    if not utility_eligible:
        return None
    from cgcr.rankers.base import validate_permutation
    validate_permutation(permutation, len(labels))
    if k < 1:
        raise ValueError('k must be positive')
    gains = DEFAULT_GAINS if gains is None else gains
    values = [float(gains[label]) for label in labels]
    if any(v < 0 or not math.isfinite(v) for v in values):
        raise ValueError('Gains must be finite and nonnegative')
    dcg = lambda v: sum(g/math.log2(i+2) for i,g in enumerate(v[:k]))
    ideal = dcg(sorted(values, reverse=True))
    return dcg([values[i] for i in permutation])/ideal if ideal else 0.


def recall_exact_at_k(labels, permutation, k=5):
    from cgcr.rankers.base import validate_permutation
    validate_permutation(permutation, len(labels))
    exact = sum(label == 'E' for label in labels)
    return sum(labels[i] == 'E' for i in permutation[:k])/exact if exact else None


def kendall_tau(first, second):
    """Tau-b between two complete permutations expressed as item indices."""
    from cgcr.rankers.base import validate_permutation
    validate_permutation(first, len(first)); validate_permutation(second, len(first))
    if len(first) < 2:
        return None
    from scipy.stats import kendalltau
    a = [first.index(i) for i in range(len(first))]
    b = [second.index(i) for i in range(len(first))]
    return float(kendalltau(a,b).statistic)
