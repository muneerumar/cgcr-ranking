"""public adaptive-attack categories and private engine loader only.

Generation code, prompts, fake reviews and attacked texts must remain in private/.
This is the project's own verifier-screened review-flooding/mimicry protocol.
"""
from pathlib import Path
import importlib.util
import hashlib

CATEGORIES=('review_flooding','verifiability_mimicry')


def load_private_engine(path,private_root,expected_sha256=None):
    path=Path(path).resolve();root=Path(private_root).resolve()
    if not path.is_relative_to(root) or path.suffix!='.py':raise ValueError('Adaptive engine must live under private/')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_sha256 and digest!=expected_sha256:raise ValueError('Private adaptive engine changed')
    spec=importlib.util.spec_from_file_location('cgcr_private_adaptive_'+digest,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if not callable(getattr(module,'run',None)):raise ValueError('Private engine must expose run(stage, context)')
    return module
