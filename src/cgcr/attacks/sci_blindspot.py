"""SCI-Defense blind-spot interfaces, arXiv:2605.21948 Section 5.

Reimplementation, not official code. Bounded target-only edits on Home & Kitchen
replace the source's Automotive/Toys experiments. Specification Amplification
introduces unsupported precision and is tagged fact_altering. Use-Case
Saturation preserves listed specifications but compatibility claims remain
unverified. Neither implements a verifier or any  work.
"""
from cgcr.attacks.base import AttackCategory


def categories():
    return [AttackCategory('sci_specification_amplification','fact_altering',
        'https://arxiv.org/abs/2605.21948','reimplementation','Section 5',
        'Home & Kitchen; bounded edits; Qwen 4-bit; unsupported precision',('SPEC',)),
        AttackCategory('sci_use_case_saturation','fact_preserving',
        'https://arxiv.org/abs/2605.21948','reimplementation','Section 5',
        'Home & Kitchen; bounded edits; Qwen 4-bit; query hidden as in source',('USE_CASE',))]
