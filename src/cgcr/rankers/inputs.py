"""common seller baseline or explicitly provisional evidence-only debug input."""
import json
from cgcr.data.evidence import canonical_title,evidence_details

INPUT_MODES=('seller_text','neutral_v0_details_only')


def render_input(row,mode,schema,max_chars=None):
    if mode=='seller_text':
        text=row['seller_text']
    elif mode=='neutral_v0_details_only':
        # No extraction/licensing exists yet; do not invent licensed variant claims.
        title=canonical_title(row['catalog'],schema)
        details=evidence_details(row['catalog'],schema)
        text=title+'\n'+'\n'.join(f'{key}: '+json.dumps(value,ensure_ascii=False) for key,value in details.items())
    else:
        raise ValueError(f'Unknown input mode: {mode}')
    truncated=max_chars is not None and len(text)>max_chars
    return (text[:max_chars] if max_chars is not None else text), truncated
