"""CORE interfaces, AmazonCOREBench CORE/core_prompts.py / demo.py.

Official source prompts are read privately for local evaluation only (upstream
states no license). Deviations: Qwen 4-bit, one accepted variant, bounded suffix,
no feedback optimizer, no source data redistribution. Exact fake-review code
and all review-style outputs are private under the disclosure policy.
"""
from cgcr.attacks.base import AttackCategory


def categories():
    return [AttackCategory('core_'+style,'fact_preserving',
        'https://github.com/haibojin001/AmazonCOREBench',
        'reimplementation from official code; evaluation only',
        'CORE/core_prompts.py; demo suffix styles',
        'Qwen 4-bit; bounded suffix; one accepted variant; no feedback optimization',
        ('SUBJECTIVE',)) for style in ('review','reasoning')]
