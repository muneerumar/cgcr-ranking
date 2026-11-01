"""outcome-independent detail masks and review-supported content proposals.

Masks affect canonical EVIDENCE detail attributes for every candidate. Price,
ratings and categories remain fixed. Proposal extraction is lexical; NLI support
is an operational evidence proxy, not proof that a customer report is true.
"""
from copy import deepcopy
import random
import re

from cgcr.data.evidence import evidence_details
from cgcr.rankers.fixed_order import stable_seed


def mask_details(row, schema, k, seed, query_id):
    if k < 0:raise ValueError('Nonnegative mask size required')
    result=deepcopy(row)
    details=evidence_details(row['catalog'],schema)
    keys=sorted(details)
    random.Random(stable_seed(seed,str(query_id),row['item_id'],'T7-details-mask')).shuffle(keys)
    kept=sorted(keys[:k])
    # Canonicalize before masking so aliases cannot leak a supposedly hidden key.
    result['catalog']['details']={key:details[key] for key in kept}
    return result,kept


def review_use_case_phrases(sentences):
    """Deterministic short activity phrases; no rank/outcome-based proposals."""
    seen=set();result=[]
    for sentence in sentences:
        text=sentence.text if hasattr(sentence,'text') else str(sentence)
        for match in re.finditer(r'\b(?:use(?:d|ful)?|using|ideal|perfect|great|suitable)\b.{0,35}?\bfor\s+([^.!?;\n]{4,100})',text,re.I):
            phrase=match.group(1).strip(' ,:')
            if len(phrase.split())>15:continue
            claim='Suitable for '+phrase.rstrip('.')+'.'
            key=' '.join(claim.casefold().split())
            if key not in seen:seen.add(key);result.append(claim)
    return result


def append_improvement(clean_text, claim):
    if not claim.strip() or claim.casefold() in clean_text.casefold():
        raise ValueError('Improvement must add a previously absent assertion')
    return clean_text+'\n'+claim
