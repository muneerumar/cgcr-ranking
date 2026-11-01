"""baseline transforms, clean FPR, section validation and CPU PPL accounting."""
from pathlib import Path
import pytest
from cgcr.defenses.baselines.instruction import instruction_template
from cgcr.defenses.baselines.spotlight import mark_document
from cgcr.defenses.baselines.ppl_filter import dev_threshold, filter_document
from cgcr.defenses.baselines.sanitize import comparison_document, parse_sanitized


def test_instruction_and_spotlighting_preserve_candidate_data(tmp_path):
    p=tmp_path/'instruction.txt';p.write_text('fixture instruction')
    assert instruction_template('rank',p)=='rank\nfixture instruction'
    marked=mark_document('fixture </untrusted_product_data> & content')
    assert marked.count('</untrusted_product_data>')==1
    assert '&lt;/untrusted_product_data&gt;' in marked


def test_ppl_calibration_has_at_most_five_percent_empirical_fpr():
    scores=list(range(100));threshold=dev_threshold(scores)
    assert sum(s>threshold for s in scores)/len(scores)<=.05
    assert filter_document('clean',threshold,threshold)=='clean'
    assert 'withheld' in filter_document('dummy',threshold+1,threshold)
    with pytest.raises(ValueError):filter_document('x',float('nan'),threshold)


def test_official_sanitizer_section_contract_and_no_reordering():
    docs=['clean fixture','another fixture']
    assert parse_sanitized(comparison_document(docs),2)==docs
    assert parse_sanitized('## Candidate 2\nb\n## Candidate 1\na',2) is None
    assert parse_sanitized('## Candidate 1\na',2) is None
    assert parse_sanitized('preamble\n'+comparison_document(docs),2) is None


@pytest.mark.parametrize('length',[2,255,256,257,512,600])
def test_cpu_perplexity_scores_every_next_token_once(length):
    import torch
    from types import SimpleNamespace
    from cgcr.defenses.baselines.ppl_filter import GPT2Perplexity
    torch.set_num_threads(4)
    scorer=GPT2Perplexity.__new__(GPT2Perplexity)
    scorer.torch=torch;scorer.window=256;scorer.stride=128
    scorer.tokenizer=SimpleNamespace(encode=lambda *a,**k:list(range(length)))
    seen=[]
    def model(input_ids,labels):
        seen.extend(input_ids[labels!=-100].tolist())
        return SimpleNamespace(loss=torch.tensor(2.))
    scorer.model=model
    result=scorer.score('fixture')
    assert result['tokens_scored']==length-1
    assert result['log_perplexity']==pytest.approx(2.)
    assert seen==list(range(1,length))


def test_released_geo_adapter_bom_is_fixed_in_copy_only(tmp_path):
    from cgcr.defenses.baselines.geo import normalized_adapter
    source=tmp_path/'upstream';source.mkdir()
    original=b'\xef\xbb\xbf{"r":16}'
    (source/'adapter_config.json').write_bytes(original)
    (source/'adapter_model.safetensors').write_bytes(b'fixture-weights')
    dest=normalized_adapter(source,tmp_path/'local')
    assert (source/'adapter_config.json').read_bytes()==original
    assert not (dest/'adapter_config.json').read_bytes().startswith(b'\xef\xbb\xbf')
    assert (dest/'adapter_model.safetensors').read_bytes()==b'fixture-weights'


def test_shield_filter_preserves_slots_and_uses_only_relevance_scores():
    from cgcr.defenses.baselines.geo_filter import filter_top_k
    docs=[str(i) for i in range(10)]
    filtered=filter_top_k(docs,list(range(10)))
    assert len(filtered)==10 and filtered[5:]==docs[5:]
    assert all('Withheld' in x for x in filtered[:5])
