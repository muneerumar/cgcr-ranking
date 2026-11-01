"""official GEO-Defender Shield scoring with released LoRA on local MPS.

Ccchi5ato/GEO-Defender 8219ff12d2496e55d76a7fd7ea83c35865aa045b, MIT.
Official score/rank methods are imported unmodified. Loader adaptation: pinned
local base/tokenizer, float16 MPS rather than CUDA bitsandbytes/device_map=auto.
Shield-only ranking is distinct from full TFSG answer generation and from Qwen.
"""
import importlib.util
import json
from pathlib import Path
import shutil
import sys


def normalized_adapter(source, destination):
    """PEFT json.load rejects upstream's UTF-8 BOM; preserve original bytes."""
    source,destination=Path(source),Path(destination)
    raw=(source/'adapter_config.json').read_bytes()
    parsed=json.loads(raw.decode('utf-8-sig'))
    destination.mkdir(parents=True,exist_ok=True)
    (destination/'adapter_config.json').write_text(json.dumps(parsed,indent=2)+'\n')
    weights=destination/'adapter_model.safetensors'
    if not weights.exists():shutil.copyfile(source/'adapter_model.safetensors',weights)
    return destination


def load_shield(upstream, base, adapter, device='mps'):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    if device not in ('cpu','mps'):raise ValueError('Local CPU/MPS only')
    if device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
    torch.set_num_threads(4)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads()>4:raise
    adapter=normalized_adapter(adapter,Path(base).parent/'geo-adapter-t7')
    path=Path(upstream)/'geo_defender/reranker.py'
    spec=importlib.util.spec_from_file_location('cgcr_official_geo_reranker',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    shield=module.ShieldReranker.__new__(module.ShieldReranker)
    shield.max_length=2048
    shield.tokenizer=AutoTokenizer.from_pretrained(str(base),local_files_only=True,padding_side='left')
    shield.tokenizer.pad_token=shield.tokenizer.eos_token
    model=AutoModelForCausalLM.from_pretrained(str(base),local_files_only=True,
        torch_dtype=torch.float16,attn_implementation='eager').to(device)
    shield.model=PeftModel.from_pretrained(model,str(adapter),is_trainable=False).eval()
    shield.prefix_tokens=shield.tokenizer.encode(module.PREFIX,add_special_tokens=False)
    shield.suffix_tokens=shield.tokenizer.encode(module.SUFFIX,add_special_tokens=False)
    shield.yes_token_id=shield.tokenizer.convert_tokens_to_ids('yes')
    shield.no_token_id=shield.tokenizer.convert_tokens_to_ids('no')
    return shield
