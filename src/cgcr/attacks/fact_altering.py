"""project-defined fact-altering control interface (not a paper reproduction).

Reimplementation / new control: private code inflates one or two numeric values
that occur in actual catalog details. Public code contains no transformation
logic, exact templates or attacked values. No claim verification is implemented.
"""
from cgcr.attacks.base import AttackCategory


def categories():
    return [AttackCategory('fact_altering_specs','fact_altering','project-defined control',
        'reimplementation / new control','T5 step 6e',
        'Inflates one or two existing numeric catalog-details values',('SPEC',))]
