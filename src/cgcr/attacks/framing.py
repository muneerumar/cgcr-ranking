"""FramingBench interfaces, Alhamoud et al., Table 1 / variant generators.

Official code: m1k2zoo/framingbench (MIT), pinned in the run manifest.
Deviations: target seller document only, bounded edits, Qwen 4-bit drafting;
M1 rewrites a bounded span and M3's preamble stays inside the target document.
Q4 uses generic synthetic authority instead of real expert endorsements.
Selection is descending Table 1 NRG, ties by technique code; no stacked methods.
"""
from cgcr.attacks.base import AttackCategory

CODES=('M1','M3','Q5','M6','Q4','Q2')
NAMES=('narrative_framing','criteria_reframing','social_proof','anchoring','expert_quotation','statistics_addition')


def categories():
    return [AttackCategory('framing_'+code+'_'+name,'fact_preserving',
            'https://github.com/m1k2zoo/framingbench','official code + bounded adaptation',
            'Table 1; variant_gen',__doc__.split('Deviations: ')[1].split('Selection is')[0].strip(),
            ('AUTHORITY',) if code=='Q4' else ('SUBJECTIVE',),code=='Q4')
            for code,name in zip(CODES,NAMES)]
