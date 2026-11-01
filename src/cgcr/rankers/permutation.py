"""seeded order ensembles using median, mean, then original-index tie breaks."""
from dataclasses import asdict, dataclass
import random
import statistics
from cgcr.rankers.base import validate_permutation


@dataclass
class PermutationResult:
    permutation: list[int]
    median_ranks: list[float]
    mean_ranks: list[float]
    orders: list[dict]


def aggregate_rankings(permutations, n):
    if not permutations:
        raise ValueError('At least one ranking is required')
    ranks = [[] for _ in range(n)]
    for permutation in permutations:
        validate_permutation(permutation, n)
        for rank, item in enumerate(permutation, 1):
            ranks[item].append(rank)
    medians = [statistics.median(values) for values in ranks]
    means = [statistics.mean(values) for values in ranks]
    return sorted(range(n), key=lambda i: (medians[i], means[i], i)), medians, means


def rank_with_permutations(ranker, query, docs, seed, orders=3):
    if orders < 1 or not docs:
        raise ValueError('Need documents and a positive order count')
    rng = random.Random(seed)
    logs, mapped = [], []
    for order in range(orders):
        shuffle = rng.sample(range(len(docs)), len(docs))
        result = ranker.rank(query, [docs[i] for i in shuffle], seed+order)
        validate_permutation(result.permutation, len(docs))
        permutation = [shuffle[i] for i in result.permutation]
        mapped.append(permutation)
        logs.append({'shuffle': shuffle, 'seed': seed+order, 'permutation_original': permutation,
                     'result': asdict(result)})
    permutation, medians, means = aggregate_rankings(mapped, len(docs))
    return PermutationResult(permutation, medians, means, logs)
