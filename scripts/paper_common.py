"""Deterministic publication rendering from frozen aggregate metrics only."""
from __future__ import annotations
import csv
import hashlib
import io
import json
import math
import re
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
import yaml

ROOT = Path(__file__).resolve().parents[1]
RUN = 'results'
CATEGORIES = {'hk_qwen_full': 'Home_and_Kitchen', 'electronics_qwen': 'Electronics', 'tools_qwen': 'Tools_and_Home_Improvement'}
CATEGORY_LABELS = {'Home_and_Kitchen': 'Home and Kitchen', 'Electronics': 'Electronics', 'Tools_and_Home_Improvement': 'Tools and Home Improvement'}
METHODS = ['no_defense', 'details_only', 'cgcr', 'instruction', 'spotlight', 'ppl_filter', 'sanitizer', 'geo_filter']
METHOD_LABELS = {'no_defense': 'No defense', 'seller_text': 'Seller text', 'details_only': 'Details only', 'cgcr': 'CGCR', 'instruction': 'Instruction', 'spotlight': 'Spotlight', 'ppl_filter': 'PPL filter', 'sanitizer': 'Sanitizer', 'geo_filter': 'Shield filter + Qwen'}
ATTACK_LABELS = {'core_reasoning_opt5': 'CORE opt5', 'core_reasoning': 'CORE', 'framing_M3_opt5': 'M3 opt5', 'framing_M3_criteria_reframing': 'M3', 'framing_Q4_expert_quotation': 'Q4', 'framing_Q5_social_proof': 'Q5'}
COMMON_ATTACKS = ['core_reasoning_opt5', 'core_reasoning', 'framing_M3_opt5']
SEED = 13
BOOTSTRAPS = 10000


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def artifact_sha(path):
    """Byte hash, except PNG: hash of decoded pixels, size and dpi.

    PNG compression bytes differ across platforms even when the rendered
    pixels are identical, so raster figures are compared by content."""
    path = Path(path)
    if path.suffix != '.png':
        return sha(path)
    from PIL import Image
    with Image.open(path) as img:
        img = img.convert('RGBA')
        dpi = tuple(round(float(v), 3) for v in img.info.get('dpi', (0, 0)))
        header = f'{img.width}x{img.height};dpi={dpi};'.encode()
        return hashlib.sha256(header + img.tobytes()).hexdigest()


def pointer(parts):
    return '/' + '/'.join(str(p).replace('~', '~0').replace('/', '~1') for p in parts) if parts else ''


def resolve(value, key):
    for part in key.split('/')[1:] if key else []:
        part = part.replace('~1', '/').replace('~0', '~')
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def bootstrap(values):
    from cgcr.metrics.attack import bootstrap_mean
    return bootstrap_mean(values, SEED, BOOTSTRAPS)


def binary_wilcoxon(left, right):
    """Exact two-sided Wilcoxon for paired binary outcomes, dropping zero ties.

    Every nonzero absolute difference is one, hence every midrank is identical.
    The exact sign-permutation Wilcoxon null is therefore a binomial distribution
    over the discordant pairs. No normal approximation or exponential enumeration.
    """
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape or not np.isin(left, [0, 1]).all() or not np.isin(right, [0, 1]).all():
        raise ValueError('Paired binary vectors of the same shape are required')
    differences = left - right; nonzero = differences[differences != 0]
    positive = int((nonzero > 0).sum()); negative = int((nonzero < 0).sum())
    statistic = min(positive, negative) * (len(nonzero) + 1) / 2
    p = float(binomtest(positive, len(nonzero), .5).pvalue) if len(nonzero) else 1.
    return dict(statistic=statistic, p_raw=p, n=len(left), discordant=len(nonzero), cgcr_only=positive, baseline_only=negative)


def holm(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind='stable'); output = np.empty(len(values)); running = 0.
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index]); output[index] = min(1., running)
    return output.tolist()



class Package:
    def __init__(self, root, bundle):
        self.root, self.run = Path(root), RUN
        self.tables, self.figures, self.notes = bundle['tables'], bundle['figures'], bundle['notes']
        name='results/publication_metrics.json'
        digest=sha(self.root/name)
        self.sources={name:dict(sha256=digest,format='json')}
        self.values={key:dict(value=record['value'], operation=record['calculation'], parents=[],
            sources=[dict(file=name,key_path=pointer(['values',key,'value']),sha256=digest)],
            upstream=record['upstream_manifest']) for key,record in bundle['values'].items()}

    def get(self,key):
        return self.values[key]['value']


def cell_value(p,cell):
    return p.get(cell['node']) if isinstance(cell,dict) and 'node' in cell else cell


def build(root=ROOT,run=RUN):
    root=Path(root)
    bundle=json.loads((root/'results/publication_metrics.json').read_text())
    if bundle['split']!='test' or bundle['scientific_freeze']!='f5bf1d5':
        raise ValueError('Unexpected frozen publication bundle')
    p=Package(root,bundle)
    # Recompute exact binary tests and each Holm family from aggregate discordant
    # counts. Previously computed query-bootstrap intervals remain unchanged.
    families={}
    for row in p.tables['T1_paired_tests']['rows']:
        get=lambda name:cell_value(p,row[name])
        positive,negative=int(get('cgcr_only')),int(get('baseline_only'))
        n=positive+negative
        statistic=min(positive,negative)*(n+1)/2
        expected=float(binomtest(positive,n,.5).pvalue) if n else 1.
        if statistic!=get('statistic') or not math.isclose(expected,get('p_raw'),abs_tol=1e-15):
            raise ValueError('Aggregate paired-test reconstruction failed')
        families.setdefault(row['category'],[]).append(row)
    for rows in families.values():
        calculated=holm([cell_value(p,r['p_raw']) for r in rows])
        if any(not math.isclose(x,cell_value(p,r['p_holm']),abs_tol=1e-15) for x,r in zip(calculated,rows)):
            raise ValueError('Holm family reconstruction failed')
    return p


def escape(value):
    return ''.join({'&': r'\&', '%': r'\%', '_': r'\_', '#': r'\#', '$': r'\$', '{': r'\{', '}': r'\}'}.get(c, c) for c in str(value))


def label(value):
    return METHOD_LABELS.get(value, ATTACK_LABELS.get(value, CATEGORY_LABELS.get(value, str(value))))


def fmt(value, digits=4):
    if value is None: return '--'
    if isinstance(value, (int, np.integer)): return str(value)
    if abs(value) < .0001 and value != 0: return f'{value:.2e}'
    return f'{value:.{digits}f}'


def interval(p, row, prefix):
    mean, lo, hi = [cell_value(p, row[prefix + '_' + f]) for f in ('mean', 'ci_low', 'ci_high')]
    return '--' if mean is None else f'{mean:.4f} [{lo:.4f}, {hi:.4f}]'


def latex_rows(p, key):
    rows = p.tables[key]['rows']; output = []; group = None
    if key in ('T1_main_robustness', 'T4_llama_transfer'):
        headers = ['Attack', 'Method', 'Promote@3 [CI]', 'n', 'NRG [CI]', 'n']; spec = '@{}llrcrcr@{}'
        # Six columns, with compact spacing; category shown as a group header.
        spec = '@{}llrcrc@{}'
        for row in rows:
            if group != row['category']:
                group = row['category']; output.append(r'\multicolumn{6}{@{}l}{\textit{' + escape(label(group)) + r'}} \\')
            output.append(' & '.join([escape(label(row['attack'])), escape(label(row['method'])), interval(p,row,'promote3'), fmt(cell_value(p,row['promote3_n'])), interval(p,row,'nrg'), fmt(cell_value(p,row['nrg_n']))]) + r' \\')
    elif key == 'T2_clean_quality':
        headers = ['Category', 'Method', 'NDCG@10 [CI]', 'n']; spec = '@{}llrc@{}'
        for row in rows: output.append(' & '.join([escape(label(row['category'])), escape(label(row['method'])), interval(p,row,'ndcg10'), fmt(cell_value(p,row['ndcg10_n']))]) + r' \\')
    elif key == 'T3_responsiveness':
        headers = ['Category', 'Method', 'Rank improves [CI]', 'n']; spec = '@{}llrc@{}'
        for row in rows: output.append(' & '.join([escape(label(row['category'])), escape(label(row['method'])), interval(p,row,'responsiveness'), fmt(cell_value(p,row['responsiveness_n']))]) + r' \\')
    elif key == 'T5_adaptive':
        headers = ['Condition', 'Method', 'Promote@3 [CI]', 'n', 'NRG [CI]', 'n', 'Licensed [CI]']; spec = '@{}llrcrcr@{}'
        for row in rows:
            condition = 'Mimicry' if row['attack']=='mimicry' else f"B={cell_value(p,row['budget'])}, " + ('VP' if row['verified_flag']=='verified' else 'unverified')
            output.append(' & '.join([escape(condition), escape(label(row['method'])), interval(p,row,'promote3'), fmt(cell_value(p,row['promote3_n'])), interval(p,row,'nrg'), fmt(cell_value(p,row['nrg_n'])), interval(p,row,'licensed') if row['budget'] is not None else '--']) + r' \\')
    elif key == 'T6a_ranking_cost':
        headers = ['Category / model', 'Method', 's/call', 'Prompt tokens', 'Output tokens', 'Logical n']; spec = '@{}llrrrr@{}'
        for row in rows:
            category = {'Home_and_Kitchen':r'H\&K', 'Electronics':'Electronics', 'Tools_and_Home_Improvement':'Tools'}[row['category']]
            output.append(' & '.join([category + (' / Llama' if row['model']=='llama_4bit' else ' / Qwen'), escape(label(row['method'])), *[fmt(cell_value(p,row[f]),2) for f in ('recorded_s_per_call','prompt_tokens_per_call','completion_tokens_per_call','logical_calls')]]) + r' \\')
    elif key == 'T6b_offline_cost':
        headers = ['Category', 'Stage', 'Items', 'Visited texts', 'Total s', 's / 1k items']; spec = '@{}llrrrr@{}'
        for row in rows: output.append(' & '.join([escape(label(row['category'])), row['phase'], *[fmt(cell_value(p,row[f]),2) for f in ('unique_items','visited_item_text_jobs','supervised_seconds','seconds_per_1000_items')]]) + r' \\')
    elif key == 'T1_paired_tests':
        headers = ['Category', 'Baseline', 'Attack', 'n', 'Delta [CI]', 'p (Holm)']; spec = '@{}lllcrr@{}'
        for row in rows: output.append(' & '.join([escape(label(row['category'])), escape(label(row['baseline'])), escape(label(row['attack'])), fmt(cell_value(p,row['n'])), interval(p,row,'delta'), fmt(cell_value(p,row['p_holm']))]) + r' \\')
    else:
        headers = ['Category', 'Method', 'Mean attack success [CI]', 'n', 'Responsiveness [CI]', 'n']; spec = '@{}llrcrc@{}'
        for row in rows: output.append(' & '.join([escape(label(row['category'])), escape(label(row['method'])), interval(p,row,'attack_success'), fmt(cell_value(p,row['attack_success_n'])), interval(p,row,'responsiveness'), fmt(cell_value(p,row['responsiveness_n']))]) + r' \\')
    return headers, spec, output


def write_tables(p, output):
    output = Path(output); folder = output/'tables'; folder.mkdir(parents=True, exist_ok=True)
    for key, table in p.tables.items():
        stream = io.StringIO(newline=''); writer = csv.writer(stream, lineterminator='\n'); writer.writerow(table['columns'])
        for row in table['rows']: writer.writerow([cell_value(p,row.get(col)) for col in table['columns']])
        (folder/(key+'.csv')).write_text(stream.getvalue())
        headers, spec, rows = latex_rows(p,key)
        text = ['% Generated by scripts/make_paper_tables.py; do not edit numbers.', r'\begingroup', r'\footnotesize\setlength{\tabcolsep}{3pt}', r'\begin{longtable}{'+spec+'}', r'\caption{'+table['caption']+r'}\label{tab:'+key+r'}\\', r'\toprule', ' & '.join(headers)+r' \\', r'\midrule', r'\endfirsthead', r'\toprule', ' & '.join(headers)+r' \\', r'\midrule', r'\endhead', r'\bottomrule', r'\endfoot', *rows, r'\end{longtable}', r'\endgroup', '']
        (folder/(key+'.tex')).write_text('\n'.join(text))
    (output/'RESULTS_README.md').write_text('# Saved-results paper package\n\nRegenerate with `python scripts/make_paper_tables.py` then `python scripts/make_paper_figures.py`. These commands perform statistical postprocessing only.\n\n'+'\n'.join('- '+note for note in p.notes)+'\n\nThe manifest resolves exported numeric cells to the released aggregate bundle. Original upstream file hashes and calculation ancestry are retained in results/upstream_provenance.json; upstream records themselves are not included. Each numeric CSV cell references a value ID, JSON pointer and SHA-256 in the released bundle. Derived values state the calculation. Figures reference the same table cells. Source SHA-256s cover whole files. Query-bootstrap intervals are frozen publication aggregates, not recomputed from withheld records; exact binary Wilcoxon and Holm tests are independently reconstructed from released discordant counts. JSONL key paths index the array of non-empty records; text-log key paths index zero-based lines; YAML paths address its parsed mapping. No attack prose, templates, reviews or claim text is exported.\n\nOnly evaluated grid cells are exported. Human-readable LaTeX combines mean and CI columns and uses compact labels; the matching CSV preserves all precision, denominators and auxiliary fields. Dimensions, tick marks, document identifiers and style constants are layout, not empirical measurements; plotting code is hashed in the manifest.\n')



def write_manifest(p,output):
    output=Path(output)
    tables={key:dict(csv='tables/'+key+'.csv',latex='tables/'+key+'.tex',
        rows=[{col:row[col]['node'] for col in table['columns'] if isinstance(row.get(col),dict) and 'node' in row[col]} for row in table['rows']])
        for key,table in p.tables.items()}
    artifacts={}
    for folder in ('tables','figures'):
        if not (output/folder).exists():continue
        for path in sorted((output/folder).iterdir()):
            if path.suffix in ('.csv','.tex','.pdf','.png'):
                artifacts[str(path.relative_to(output))]=dict(sha256=artifact_sha(path))
    code={name:dict(sha256=sha(p.root/name)) for name in
          ('scripts/paper_common.py','scripts/make_paper_tables.py','scripts/make_paper_figures.py')}
    manifest=dict(schema_version=2,study_run='20261028-011906_final_study',split='test',scientific_freeze='f5bf1d5',
        statistical_unit='query',notes=p.notes,source_files=p.sources,values=p.values,tables=tables,
        figures=p.figures,artifacts=artifacts,generator_sources=code,
        upstream_provenance=dict(file='results/upstream_provenance.json',sha256=sha(p.root/'results/upstream_provenance.json')))
    dump(output/'results_manifest.json',manifest)
    return manifest
