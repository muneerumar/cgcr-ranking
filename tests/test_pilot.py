"""query-level utility flags, dev-only stratified sampling and fixture outputs."""
import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cgcr.data.candidate_sets import annotate_utility, select_pilot, utility_eligibility, build_candidates


def test_utility_definition_and_stratified_determinism():
    rows = [{'query_id': str(q), 'esci_label': 'S' if q < 600 and i == 0 else 'E'}
            for q in range(900) for i in range(10)]
    flags = utility_eligibility(rows)
    selected = select_pilot(flags, flags, 13)
    assert selected == select_pilot(reversed(list(flags)), flags, 13)
    assert len(selected) == 300 and len(set(selected)) == 300
    assert sum(flags[q] for q in selected) >= 200
    assert any(not flags[q] for q in selected)
    assert selected != select_pilot(flags, flags, 14)
    for row in annotate_utility(rows):
        assert row['utility_eligible'] == (int(row['query_id']) < 600)
    with pytest.raises(ValueError, match='eligible'):
        select_pilot(flags, {q: False for q in flags})
    with pytest.raises(ValueError, match='insufficient'):
        select_pilot(['q'], {'q': True})
    with pytest.raises(ValueError, match='Missing'):
        select_pilot(['a', 'b'], {'a': True}, size=2, min_utility=1)


def test_build_candidates_exposes_utility():
    from test_data import pool_fixture
    pool, catalog = pool_fixture()
    assert all(r['utility_eligible'] for r in build_candidates('q', pool, catalog))
    for row in pool:
        row['esci_label'] = 'E'
    assert not any(r['utility_eligible'] for r in build_candidates('q', pool, catalog))


def test_full_evidence_pipeline_fixture(tmp_path, monkeypatch):
    import yaml
    from cgcr.data.amazon_reviews_2023 import CATALOG_SCHEMA
    root = Path(__file__).parents[1]
    spec = importlib.util.spec_from_file_location('prepare_evidence_test', root / 'scripts/prepare_evidence.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    out = tmp_path / 'data/processed/Home_and_Kitchen'; out.mkdir(parents=True)
    (tmp_path / 'configs/schemas').mkdir(parents=True)
    (tmp_path / 'configs/schemas/Home_and_Kitchen_synonyms.yaml').write_text('synonyms: {}\nexclude: {}\n')
    rows = [{'query_id': str(q), 'item_id': f'P{i}', 'product_id': f'C{i}',
             'esci_label': 'S' if q < 4 and i == 9 else 'E', 'seller_text': 'Color Red',
             'catalog': {'title': 'prose', 'features': ['prose'], 'details': '{"Color":"Red"}',
                         'price': '10', 'average_rating': 4., 'rating_number': 5, 'categories': ['Kitchen'], 'store': ''},
             'reviews': f'reviews/P{i}.parquet'} for q in range(8) for i in range(10)]
    schema = pa.schema([(k, pa.string()) for k in ('query_id', 'item_id', 'product_id', 'esci_label', 'seller_text')]
                       + [('catalog', pa.struct(list(CATALOG_SCHEMA)[1:])), ('reviews', pa.string())])
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), out / 'candidates.parquet')
    (out / 'split_ids.json').write_text(json.dumps({'query_ids': {'dev': ['0','1','2','3','4','5'], 'test': ['6','7']}}))
    # Prove there are no network calls anywhere in this preparation.
    import urllib.request
    monkeypatch.setattr(urllib.request, 'urlopen', lambda *a, **k: pytest.fail('Network forbidden'))
    config = {'seed': 13, 't3': {'schema_min_coverage': .02, 'pilot_queries': 3, 'min_utility_queries': 2}}
    run, stats = module.run_pipeline(tmp_path, config)
    ids = json.loads((out / 'pilot_query_ids.json').read_text())['query_ids']
    assert len(ids) == 3 and set(ids) <= {'0','1','2','3','4','5'}
    assert stats['pilot']['utility_eligible'] >= 2
    evidence = pq.read_table(out / 'evidence.parquet').to_pylist()
    assert len(evidence) == 80 and all(r['joined_via_variant'] for r in evidence)
    assert not {'features', 'title', 'store'} & evidence[0]['catalog'].keys()
    assert pq.read_table(out / 'candidates.parquet').schema.field('utility_eligible').type == pa.bool_()
    assert pq.read_table(out / 'pilot.parquet').num_rows == 30
    generated = yaml.safe_load((tmp_path / 'configs/schemas/Home_and_Kitchen.yaml').read_text())
    assert [a['canonical_name'] for a in generated['attributes']] == ['color']
    assert stats['utility_by_split']['test'] == {'queries': 2, 'utility_eligible': 0, 'all_exact': 2}
    for name in ('git.txt', 'env.txt', 'config.yaml', 'seeds.yaml', 'metrics.json', 'per_query.jsonl', 'run.log', 'code_manifest.json'):
        assert (run / name).exists()
