"""paired target positions; the non-target order never changes by condition."""
import hashlib
import random


def stable_seed(seed, *keys):
    return int.from_bytes(hashlib.sha256('\0'.join(map(str, (seed,)+keys)).encode()).digest()[:4], 'big')


def fixed_order(n, target, seed, query_id, slot=10):
    if n != 10 or not 0 <= target < n or slot not in (1, 5, 10):
        raise ValueError('Protocol requires ten candidates and slot 1, 5 or 10')
    others = [i for i in range(n) if i != target]
    random.Random(stable_seed(seed, query_id, target, 'non-target-order')).shuffle(others)
    others.insert(slot-1, target)
    return others


def subset_ids(ids, count, seed, name):
    ids = list(ids)
    if count > len(ids):
        raise ValueError('Subset larger than population')
    return sorted(random.Random(stable_seed(seed, name)).sample(sorted(ids), count))


def prompt_prefix(ranker, query, docs):
    """Return exactly the user-message text preceding the last document body."""
    before, _ = ranker.template.split('{candidates_block}')
    return before.format(query=query, n=len(docs)) + '\n\n'.join(
        [f'[{i+1}] {doc}' for i, doc in enumerate(docs[:-1])] + [f'[{len(docs)}] '])
