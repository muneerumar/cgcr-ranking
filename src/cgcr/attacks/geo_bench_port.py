"""GEO-Bench zero-shot interface, arXiv:2605.29107 Appendix B/D.

Reimplementation: Qwen 4-bit rather than the paper's attack model; bounded
seller-only span rewrite, one accepted variant, no white-box optimization.
Private code owns the generation prompt and rewrite implementation.
"""
from cgcr.attacks.base import AttackCategory


def categories():
    return [AttackCategory('geo_zero_shot','fact_preserving',
        'https://arxiv.org/abs/2605.29107','reimplementation','Appendix B/D',
        'Qwen 4-bit; bounded span instead of whole document; one accepted variant',('SUBJECTIVE',))]
