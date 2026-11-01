#!/usr/bin/env python3
"""audit evidence, mine details-only schema, annotate utility, sample dev pilot."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import sys

import pyarrow as pa
import pyarrow.parquet as pq

from cgcr.data.candidate_sets import annotate_utility, select_pilot, utility_eligibility
from cgcr.data.evidence import EVIDENCE, evidence_record, overlap_audit
from cgcr.data.schema_builder import KeyPolicy, build_schema, variant_sensitive_keys
from cgcr.utils.io import read_yaml, write_yaml, write_jsonl
from cgcr.utils.logging import start_run, get_logger

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.tmp')
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_parquet(path, table):
    path = Path(path)
    temporary = path.with_name('.' + path.name + '.tmp')
    try:
        pq.write_table(table, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _cell(value):
    return str(value).replace('|', '\\|').replace('\n', ' ')


def schema_table(schema):
    lines = ['| Attribute | Items | Coverage % | Type | Observed units | Variant risk | Top examples |',
             '|---|---:|---:|---|---|---|---|']
    for attr in schema['attributes'][:40]:
        examples = '; '.join(v['value'] for v in attr['top_10_example_values'][:3])
        lines.append('| ' + ' | '.join(map(_cell, [attr['canonical_name'], attr['item_count'],
                     f"{attr['coverage_pct']:.4f}", attr['value_type'], ', '.join(attr['observed_units']),
                     attr['variant_sensitive'], examples])) + ' |')
    return '\n'.join(lines)


def run_pipeline(root, config, category='Home_and_Kitchen'):
    root = Path(root).resolve()
    out = root / 'data/processed' / category
    synonym_path = root / 'configs/schemas' / f'{category}_synonyms.yaml'
    synonyms = read_yaml(synonym_path)
    resolved = dict(config) | {'category': category, 'schema_scope': 'unique T2 selected parents',
                              'synonyms': synonyms, 'evidence_policy_version': 'T3-v1'}
    run = start_run('T3_evidence_schema_pilot', resolved, root / 'results')
    logger = get_logger()
    write_yaml(run / 'seeds.yaml', {'pilot': config['seed']})
    code_paths = ['scripts/prepare_evidence.py', 'src/cgcr/data/evidence.py',
                  'src/cgcr/data/schema_builder.py', 'src/cgcr/data/units.py',
                  'src/cgcr/data/candidate_sets.py', 'src/cgcr/utils/io.py',
                  'src/cgcr/utils/logging.py']
    sources = {}
    for relative in code_paths:
        source = ROOT / relative
        target = run / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        sources[relative] = sha256(source)
    save_json(run / 'code_manifest.json', sources)
    save_json(run / 'config_hash.json', {'config.yaml': sha256(run / 'config.yaml')})
    input_paths = [out / 'candidates.parquet', out / 'split_ids.json', synonym_path]
    save_json(run / 'input_manifest.json', {str(p.relative_to(root)): sha256(p) for p in input_paths})
    original = pq.read_table(out / 'candidates.parquet')
    rows = original.to_pylist()
    if not rows:
        raise ValueError('Empty candidate table')
    groups = defaultdict(list)
    for row in rows:
        groups[row['query_id']].append(row)
    for query, group in groups.items():
        if len(group) != 10 or len({r['item_id'] for r in group}) != 10:
            raise ValueError(f'Expected ten distinct items: {query}')
        if any(r['esci_label'] not in {'E', 'S', 'C', 'I'} for r in group):
            raise ValueError('Unknown ESCI label')
    items = {r['item_id']: r['catalog'] for r in rows}
    schema = build_schema(items, synonyms, category, config['t3']['schema_min_coverage'])
    policy = KeyPolicy(synonyms)
    overlap, overlap_rows = overlap_audit(rows)
    write_jsonl(run / 'per_item.jsonl', overlap_rows)
    eligibility = utility_eligibility(rows)
    splits = json.loads((out / 'split_ids.json').read_text())['query_ids']
    dev, test = set(splits['dev']), set(splits['test'])
    if dev & test or dev | test != groups.keys():
        raise ValueError('Split IDs must partition candidate query IDs')
    pilot_ids = select_pilot(dev, eligibility, seed=config['seed'],
                            size=config['t3']['pilot_queries'], min_utility=config['t3']['min_utility_queries'])
    pilot_set = set(pilot_ids)
    pilot_rows = [r for r in rows if r['query_id'] in pilot_set]
    sensitive = Counter()
    for catalog in items.values():
        sensitive.update(variant_sensitive_keys(catalog['details'], policy))
    evidence = [evidence_record(row, policy, schema) for row in rows]
    evidence_schema = pa.schema([(key, pa.string()) for key in ('query_id', 'item_id', 'product_id')]
        + [('joined_via_variant', pa.bool_()), ('variant_sensitive_keys', pa.list_(pa.string())),
           ('catalog', pa.struct([field for field in original.schema.field('catalog').type
                                 if field.name in EVIDENCE['catalog']])), ('reviews', pa.string())])
    variant_items = {r['item_id'] for r in evidence if r['joined_via_variant']}
    summary = {
        'category': category, 'run': str(run.relative_to(root)), 'seed': config['seed'],
        'overlap': overlap,
        'variant_risk': {'candidate_rows': len(rows), 'joined_via_variant_rows': sum(r['joined_via_variant'] for r in evidence),
                         'unique_items_with_any_variant_join': len(variant_items),
                         'sensitive_keys_item_counts': dict(sorted(sensitive.items()))},
        'schema': {'items': len(items), 'attributes_retained': len(schema['attributes']),
                   'min_coverage_pct': schema['min_coverage_pct'],
                   'below_threshold_attributes': len(schema['below_threshold']),
                   'excluded_observed_keys': schema['excluded_observed_keys']},
        'utility_by_split': {name: {'queries': len(ids), 'utility_eligible': sum(eligibility[q] for q in ids),
                                    'all_exact': sum(not eligibility[q] for q in ids)} for name, ids in splits.items()},
        'pilot': {'queries': len(pilot_ids), 'candidates': len(pilot_rows),
                  'utility_eligible': sum(eligibility[q] for q in pilot_ids),
                  'all_exact': sum(not eligibility[q] for q in pilot_ids),
                  'unique_items': len({r['item_id'] for r in pilot_rows}),
                  'labels': dict(sorted(Counter(r['esci_label'] for r in pilot_rows).items())),
                  'per_set_label_distribution': dict(sorted(Counter(json.dumps(dict(sorted(Counter(r['esci_label'] for r in groups[q]).items())), sort_keys=True) for q in pilot_ids).items())),
                  'sampling': 'seeded proportional eligibility strata with configured lower bound; dev only'},
    }
    # Validate all selections before mutating existing candidate artifacts.
    annotated = annotate_utility(rows)
    annotated_table = original
    if 'utility_eligible' in annotated_table.column_names:
        annotated_table = annotated_table.drop(['utility_eligible'])
    annotated_table = annotated_table.append_column('utility_eligible', pa.array([r['utility_eligible'] for r in annotated], type=pa.bool_()))
    save_parquet(out / 'candidates.parquet', annotated_table)
    for name, ids in splits.items():
        selected_ids = set(ids)
        save_parquet(out / f'{name}.parquet', annotated_table.filter(pa.array([r['query_id'] in selected_ids for r in rows])))
    save_parquet(out / 'pilot.parquet', annotated_table.filter(pa.array([r['query_id'] in pilot_set for r in rows])))
    save_parquet(out / 'evidence.parquet', pa.Table.from_pylist(evidence, schema=evidence_schema))
    save_json(out / 'pilot_query_ids.json', {'category': category, 'seed': config['seed'],
              'source_split': 'dev', 'sampling': summary['pilot']['sampling'], 'query_ids': pilot_ids})
    write_yaml(root / 'configs/schemas' / f'{category}.yaml', schema)
    write_jsonl(run / 'per_query.jsonl', ({'query_id': q, 'split': 'dev' if q in dev else 'test',
                 'utility_eligible': eligibility[q], 'pilot': q in pilot_set,
                 'labels': dict(Counter(r['esci_label'] for r in groups[q]))} for q in sorted(groups)))
    save_json(run / 'metrics.json', summary)
    save_json(out / 't3_summary.json', summary)
    save_json(out / 'schema_audit.json', schema)
    output_paths = [out / name for name in ('candidates.parquet', 'dev.parquet', 'test.parquet', 'pilot.parquet',
                   'pilot_query_ids.json', 'evidence.parquet', 't3_summary.json', 'schema_audit.json')]
    output_paths += [root / 'configs/schemas' / f'{category}.yaml']
    save_json(run / 'output_manifest.json', {str(p.relative_to(root)): sha256(p) for p in output_paths})
    evidence_doc = root / 'docs/evidence_policy.md'
    marker = '<!-- T3_MEASUREMENTS -->'
    if evidence_doc.exists():
        policy_text = evidence_doc.read_text().split(marker)[0]
        evidence_doc.write_text(policy_text + marker + '\n\nGenerated by `scripts/prepare_evidence.py`; run `' + summary['run']
                                + '`.\n\n```json\n' + json.dumps(overlap, indent=2) + '\n```\n')
    (run / 'schema_table.md').write_text(schema_table(schema) + '\n')
    logger.info('T3 complete: %s attributes, %s dev pilot queries, %s unique pilot items',
                len(schema['attributes']), len(pilot_ids), summary['pilot']['unique_items'])
    print(json.dumps(summary, indent=2))
    return run, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--category', default='Home_and_Kitchen')
    parser.add_argument('--config', default='configs/cgcr_default.yaml')
    args = parser.parse_args()
    run_pipeline(ROOT, read_yaml(ROOT / args.config), args.category)


if __name__ == '__main__':
    main()
