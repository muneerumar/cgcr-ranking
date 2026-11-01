"""public interfaces tested only with dummy private implementations."""
from pathlib import Path
import pytest
from cgcr.attacks import categories
from cgcr.attacks.base import private_path,load_private_provider
from cgcr.metrics.attack import case_metrics,bootstrap_mean,paired_position_difference


def test_registry_provenance_and_synthetic_authority():
    registry=categories()
    assert len(registry)==12 and len({c.name for c in registry})==12
    assert all(c.provenance and c.section and c.deviations for c in registry)
    q4=next(c for c in registry if 'Q4' in c.name)
    assert q4.synthetic_attribution and q4.attack_class=='fact_preserving' and q4.claim_types==('AUTHORITY',)
    assert all(c.attack_class in ('fact_preserving','fact_altering') for c in registry)


def test_private_loader_and_escape(tmp_path):
    root=tmp_path/'private';root.mkdir()
    (root/'dummy.py').write_text('def generate(**kwargs):\n    return {"dummy": True}\n')
    provider,digest=load_private_provider(root,'dummy.py')
    assert provider.generate()=={'dummy':True} and len(digest)==64
    with pytest.raises(ValueError):private_path(root,'../outside.py')
    (root/'link.py').symlink_to(tmp_path/'outside.py')
    with pytest.raises(ValueError):private_path(root,'link.py')
    with pytest.raises(ValueError):private_path(tmp_path,'dummy.py')


def test_metrics_and_unconditional_denominator():
    assert case_metrics(5,1)['NRG']==1
    assert case_metrics(5,3)['NRG']==.5
    assert case_metrics(1,4)['NRG'] is None
    assert case_metrics(4,2)['Promote@5']==0
    assert case_metrics(7,4)['Promote@5']==1
    assert case_metrics(1,1)['Success@1']==1
    with pytest.raises(ValueError):case_metrics(0,1)
    assert bootstrap_mean([0,1,None],13,1000)==bootstrap_mean([0,1,None],13,1000)
    assert bootstrap_mean([None])['n']==0


def test_paired_ci_preserves_pairing_and_undefined_mask():
    left=[{'query_id':i,'clean_rank':5,'attacked_rank':2} for i in range(5)]
    right=[dict(r,attacked_rank=3) for r in left]
    result=paired_position_difference(left,right,13,100)
    assert result['NRG']=={'mean':.25,'ci95':[.25,.25],'n':5}
    with pytest.raises(ValueError):paired_position_difference(left,right[:-1])
