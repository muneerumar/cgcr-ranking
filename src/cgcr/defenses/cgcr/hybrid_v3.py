"""extractor v3: frozen v2 SPEC proposals with numeric and phrase quality guards."""
from dataclasses import asdict
import re,time
from cgcr.defenses.cgcr.claim_quality import words,malformed_numeric,fragment_flags
from cgcr.defenses.cgcr.hybrid_extract import HybridExtractor,deterministic_specs,parse_non_spec,deduplicate_spans
from cgcr.defenses.cgcr.extract import text_sha

CUES={
 'USE_CASE':r'\b(for|use[ds]?|using|suitable|ideal|designed|intended|works|compatible)\b',
 'AUTHORITY':r'\b(recommend\w*|endorse\w*|certif\w*|test\w*|approv\w*|expert\w*|award\w*|research\w*)\b',
 'TEMPORAL':r'\b(new\w*|recent\w*|latest|updated|year|month|today|released|collection)\b'}


def non_spec_reason(kind,span):
    if len(words(span))<2:return 'fewer_than_two_words'
    flags=fragment_flags(dict(span=span,value=span))
    if flags:return flags[0]
    if kind in CUES and not re.search(CUES[kind],span,re.I):return 'no_type_cue_product_name_or_size'
    return None


class HybridV3Extractor(HybridExtractor):
    def extract(self,item_id,text,catalog,seed):
        started=time.perf_counter();audit={};original=deterministic_specs(text,catalog,self.schema,audit)
        rejected=[dict(claim=c,reason='malformed_numeric') for c in original if malformed_numeric(c['value'])]
        specs=[c for c in original if not malformed_numeric(c['value'])];deterministic_seconds=time.perf_counter()-started
        output=self.backend.generate([dict(role='system',content=self.prompt),dict(role='user',content=text)],seed=seed,max_tokens=256,temperature=0.)
        parsed,errors=parse_non_spec(output.text,text,output.completion_tokens,audit);accepted=[]
        for c in parsed:
            reason=non_spec_reason(c['claim_type'],c['span'])
            if reason:rejected.append(dict(claim=c,reason=reason))
            else:accepted.append(c)
        claims=deduplicate_spans(specs+accepted,audit,'cross_stage')
        return dict(extractor_family='v3-hybrid',extractor_version=self.version,item_id=item_id,text_sha256=text_sha(text),seed=seed,
            status='partial_failure' if errors else 'ok',claims=claims,postprocess=audit,
            quality_rejections=rejected,rejected_claims=[dict(reason_code=e) for e in errors],attempts=[asdict(output)],
            deterministic_seconds=deterministic_seconds,elapsed_s=time.perf_counter()-started,
            deterministic_spec_claims=len(specs),unmapped_spec_claims=sum(c.get('unmapped',False) for c in specs),non_spec_claims=len(accepted))
