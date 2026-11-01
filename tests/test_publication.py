"""Public aggregate provenance, paired statistics and byte-identical rendering."""
import importlib
import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]


def module(name):
    if str(ROOT/'scripts') not in sys.path:sys.path.insert(0,str(ROOT/'scripts'))
    return importlib.import_module(name)


def test_exact_binary_tests_and_holm():
    common=module('paper_common')
    value=common.binary_wilcoxon([1,1,1,0,0],[0,0,0,1,0])
    assert value['discordant']==4 and value['statistic']==2.5 and value['p_raw']==.625
    assert common.binary_wilcoxon([0,0],[0,0])['p_raw']==1
    assert common.holm([.01,.04,.03])==pytest.approx([.03,.06,.06])
    with pytest.raises(ValueError):common.binary_wilcoxon([.5],[0])


def test_aggregate_bundle_has_no_query_vectors():
    bundle=json.loads((ROOT/'results/publication_metrics.json').read_text())
    assert bundle['split']=='test' and bundle['scientific_freeze']=='f5bf1d5'
    assert all(v['value'] is None or isinstance(v['value'],(int,float)) for v in bundle['values'].values())
    def walk(value):
        if isinstance(value,dict):
            assert not {'attacked_text','raw_output','fake_review_text','query_ids','records'}&value.keys()
            for v in value.values():walk(v)
        elif isinstance(value,list):
            for v in value:walk(v)
    for path in (ROOT/'results').rglob('*.json'):
        if path.name=='upstream_provenance.json':continue
        walk(json.loads(path.read_text()))


def test_artifacts_regenerate_identically_and_provenance_resolves(tmp_path):
    common=module('paper_common');figures=module('make_paper_figures')
    first,second=tmp_path/'first',tmp_path/'second'
    p=figures.generate(ROOT,first);figures.generate(ROOT,second)
    files=lambda root:{str(f.relative_to(root)):common.artifact_sha(f) for f in root.rglob('*') if f.is_file()}
    assert files(first)==files(second)
    checked=files(first)
    # Running the table exporter again must preserve figure manifest entries.
    common.write_tables(p,first);common.write_manifest(p,first)
    assert files(first)==checked
    manifest=json.loads((first/'results_manifest.json').read_text())
    sources={name:json.loads((ROOT/name).read_text()) for name in manifest['source_files']}
    for name,record in manifest['source_files'].items():assert common.sha(ROOT/name)==record['sha256']
    for node in manifest['values'].values():
        assert node['sources'] and not node['parents']
        for ref in node['sources']:
            assert ref['sha256']==manifest['source_files'][ref['file']]['sha256']
            assert common.resolve(sources[ref['file']],ref['key_path'])==node['value']
    for name,record in manifest['artifacts'].items():
        assert common.artifact_sha(first/name)==record['sha256']
        assert common.artifact_sha(ROOT/'paper'/name)==common.artifact_sha(first/name),name
    assert (ROOT/'paper/results_manifest.json').read_bytes()==(first/'results_manifest.json').read_bytes()
    assert len(p.tables['T1_main_robustness']['rows'])==78
    assert len(p.tables['T1_paired_tests']['rows'])==66
    assert all(r['attacks'].split(';')==common.COMMON_ATTACKS for r in p.tables['F3_common_attacks_data']['rows'])
    from PIL import Image
    for name,spec in manifest['figures'].items():
        with Image.open(first/'figures'/(name+'.png')) as img:
            assert img.width==round(spec['width_in']*300)
            assert abs(img.info['dpi'][0]-300)<.1
        assert b'/Subtype /Image' not in (first/'figures'/(name+'.pdf')).read_bytes()


def test_booktabs_column_counts():
    import re
    for path in (ROOT/'paper/tables').glob('*.tex'):
        text=path.read_text()
        spec=re.search(r'\\begin\{longtable\}\{([^\n]+)\}',text)[1]
        count=sum(spec.count(c) for c in 'lcr')
        for line in text.splitlines():
            if line.endswith(r' \\') and not line.startswith(r'\multicolumn'):
                assert len(re.findall(r'(?<!\\)&',line))+1==count


def test_figure_layouts_use_paired_values_compact_groups_and_plain_legends(monkeypatch,tmp_path):
    common=module('paper_common');figures=module('make_paper_figures')
    from matplotlib import colors
    from matplotlib.lines import Line2D
    p=common.build(ROOT,common.RUN);captured={}
    monkeypatch.setattr(figures,'save',lambda p,fig,folder,name,rows,caption:captured.update({name:(fig,rows)}))
    with figures.plt.rc_context(figures.STYLE):
        figures.grouped(p,tmp_path,'F2_strongest_attack','T1_main_robustness','promote3','fixture','core_reasoning_opt5')
        figures.quality(p,tmp_path);figures.flooding(p,tmp_path)
    f2,rows=captured['F2_strongest_attack'];ax=f2.axes[0]
    assert len(ax.patches)==18 and len(ax.texts)==18
    chosen=[(i,r) for i,r in enumerate(p.tables['T1_main_robustness']['rows']) if r['attack']=='core_reasoning_opt5']
    positions=figures.category_positions(chosen)
    for category,expected in zip(figures.CATS,(8,5,5)):
        selected=[(x,i,r) for x,i,r in positions if r['category']==category]
        assert len(selected)==expected
        xs=[x for x,i,r in selected]
        assert all(abs(b-a-.092)<1e-10 for a,b in zip(xs,xs[1:]))
        assert [r['method'] for x,i,r in selected]==[m for m in common.METHODS if any(r['method']==m for x,i,r in selected)]
    for bar,label,(x,index,row) in zip(ax.patches,ax.texts,positions):
        assert bar.get_height()==figures.val(p,row,'promote3_mean')
        assert label.get_text()==f"{bar.get_height():.2f}"
        assert colors.to_hex(bar.get_facecolor())==figures.COLORS[row['method']].lower()
    assert f2.legends[0]._ncols==4
    assert len(f2.legends[0].get_texts())<=8 and ax.get_position().y1>=.83
    f4,rows=captured['F4_clean_quality'];quality_ax=f4.axes[0]
    assert not quality_ax.patches and len(quality_ax.containers)==18
    assert quality_ax.get_ylim()==pytest.approx((.70,.95))
    assert [t.get_text() for t in f4.legends[0].get_texts()]==[t.get_text() for t in f2.legends[0].get_texts()]
    f5,rows=captured['F5_review_flooding']
    assert all(a.get_ylim()==pytest.approx((0,.30)) for a in f5.axes)
    assert all('n=48' in [t.get_text() for t in a.texts] for a in f5.axes)
    handles=f5.legends[0].legend_handles
    assert all(isinstance(h,Line2D) for h in handles)
    assert [h.get_marker() for h in handles]==['o','s']
    assert all(h.get_linestyle()=='None' for h in handles)
    assert all(p.get(node)==48 for node in p.figures['F5_review_flooding']['annotations'][0]['value_nodes'])
    for fig,rows in captured.values():figures.plt.close(fig)
