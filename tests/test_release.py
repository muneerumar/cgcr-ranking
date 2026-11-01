"""Release dependency pins, public imports, data checksums and dry download plan."""
import hashlib
import importlib
import json
from pathlib import Path
import pkgutil
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]


def script(name):
    if str(ROOT/'scripts') not in sys.path:sys.path.insert(0,str(ROOT/'scripts'))
    return importlib.import_module(name)


def test_import_all_public_modules_without_loading_models():
    import cgcr
    for entry in pkgutil.walk_packages(cgcr.__path__,prefix='cgcr.'):
        assert importlib.import_module(entry.name).__doc__,entry.name


def test_frozen_dependency_versions_and_metadata():
    import tomllib
    from importlib.metadata import version
    from packaging.requirements import Requirement
    config=tomllib.loads((ROOT/'pyproject.toml').read_text())
    assert config['project']['version']=='1.0.0'
    assert 'gpu' not in config['project']['optional-dependencies']
    pins={Requirement(line).name:Requirement(line) for line in (ROOT/'requirements.lock').read_text().splitlines()
          if line and not line.startswith('#')}
    for name in ('torch','transformers','numpy','matplotlib'):
        assert version(name) in pins[name].specifier
    import yaml
    authors=yaml.safe_load((ROOT/'CITATION.cff').read_text())['authors']
    assert len(authors)==2
    assert authors[0]['email']=='muneer.umar@kust.edu.pk'
    assert authors[1]['email']=='salmaperveen89@gmail.com'


def test_data_download_is_dry_by_default_and_destinations_are_checked(monkeypatch,capsys,tmp_path):
    m=script('download_data')
    monkeypatch.setattr(m,'download',lambda *a,**k:pytest.fail('Dry plan attempted download'))
    assert m.main([])==0
    plan=json.loads(capsys.readouterr().out)
    assert not plan['network_requested']
    assert plan['total_bytes']==sum(f['content_length'] for f in plan['files'])
    assert len(plan['files'])==8
    for item in plan['files']:assert len(item['sha256'])==64


def test_downloader_reuses_only_hash_valid_files_and_rejects_escape(tmp_path):
    m=script('download_data');path=tmp_path/'data/raw/tiny.bin'
    path.parent.mkdir(parents=True);path.write_bytes(b'fixture')
    item=dict(path='data/raw/tiny.bin',content_length=7,sha256=hashlib.sha256(b'fixture').hexdigest())
    assert m.download(item,tmp_path)==path
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='integrity'):m.download(item,tmp_path)
    with pytest.raises(ValueError,match='beneath'):m.download(item|dict(path='../escape'),tmp_path)
