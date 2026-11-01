"""public category registry; exact attack material is loaded from private/."""
from cgcr.attacks import framing,core_attacks,geo_bench_port,sci_blindspot,fact_altering


def categories():
    return [c for module in (framing,core_attacks,geo_bench_port,sci_blindspot,fact_altering)
            for c in module.categories()]
