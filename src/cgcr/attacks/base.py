"""public attack metadata and private implementation loader; no payloads.

Published methods are adapted to this project's seller-only character budget.
Exact generators, templates, raw responses and attacked texts remain private.
"""
from dataclasses import dataclass, asdict
import hashlib
import importlib.util
from pathlib import Path
import sys


@dataclass(frozen=True)
class AttackCategory:
    name: str
    attack_class: str
    source: str
    provenance: str
    section: str
    deviations: str
    claim_types: tuple[str,...] = ()
    synthetic_attribution: bool = False

    def metadata(self):
        return asdict(self)


def private_path(root, relative):
    root=Path(root).resolve()
    path=(root/relative).resolve()
    if root.name!='private' or not path.is_relative_to(root):
        raise ValueError('Attack material must stay beneath private/')
    return path


def load_private_provider(root, relative='t5/provider.py'):
    path=private_path(root,relative)
    if path.suffix!='.py':raise ValueError('Private provider must be Python')
    identity=hashlib.sha256(path.read_bytes()).hexdigest()
    spec=importlib.util.spec_from_file_location('_cgcr_private_'+identity,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    if not callable(getattr(module,'generate',None)):
        raise TypeError('Private provider must expose generate()')
    return module,identity


def validate_generation(clean, edits, budget):
    """Only edit format, nonempty change and budget determine acceptance.

    No ranks, model scores, effectiveness values or factuality labels are inputs.
    Exact attack construction remains in the private provider.
    """
    attacked,stats=budget.apply(clean,edits)
    if not attacked.strip() or attacked==clean:raise ValueError('Empty or unchanged generated edit')
    return attacked,stats
