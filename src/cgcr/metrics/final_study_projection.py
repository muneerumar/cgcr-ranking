"""Planning revision: explicit grids and measured-unit compute costs, no test I/O.

All probabilities, timings and throughput rates are supplied from dev logs.
Logical cells stay in the plan even when byte-identical inputs can reuse a result.
"""
from copy import deepcopy
from collections import defaultdict

HK='Home_and_Kitchen'
CATEGORIES=(HK,'Electronics','Tools_and_Home_Improvement')
REDUCED_ATTACKS=('core_reasoning_opt5','core_reasoning','framing_M3_opt5')
REDUCED_DEFENSES=('no_defense','details_only','cgcr','spotlight','sanitizer')
TRANSFER_DEFENSES=('no_defense','details_only','cgcr','spotlight')


def revised_config(original):
    """The human-approved scope request, awaiting approval to execute."""
    cfg=deepcopy(original)
    if len(cfg['defenses'])!=8 or len(cfg['attacks']['standard_six'])!=6:raise ValueError('Expected the frozen 8 × 6 method registry')
    cfg.update(scope_revision='2026-10-28-reduced-grid',study_started=False,
        scope='Revised planned test study; execution awaits explicit user approval',
        grid_definition='execution_plan is authoritative; top-level defenses, attacks and models are registries, not a Cartesian product',
        approval=dict(required=True,status='awaiting_user_approval',approved=False,auto_start=False))
    cfg['execution_plan']=[
        dict(id='hk_qwen_full',category=HK,model='qwen_4bit',queries=147,orders=1,include_clean=True,defenses=list(cfg['defenses']),attacks=list(cfg['attacks']['standard_six']),
            adaptive=dict(queries=48,subset_of='hk_qwen_full',selection='seeded from the 147 accepted paired queries, not ranking outcomes',
                mimicry_defenses=list(cfg['defenses']),flooding_defenses=['cgcr'],budgets=[0,3,10,20],verified_modes=[False,True],
                definition='unsupported by genuine evidence at B=0; select only on genuine-review support below tau; never ranking outcomes; no padding',
                secondary='catalog-contradicted subset separately, with its observed count'),commit='Study: Home and Kitchen Qwen full'),
        *[dict(id=stage,category=category,model='qwen_4bit',queries=147,orders=1,include_clean=True,defenses=list(REDUCED_DEFENSES),attacks=list(REDUCED_ATTACKS),
            commit='Study: '+category+' Qwen') for stage,category in [('electronics_qwen','Electronics'),('tools_qwen','Tools_and_Home_Improvement')]],
        dict(id='hk_llama_transfer',category=HK,model='llama_4bit',queries=147,orders=1,include_clean=True,defenses=list(TRANSFER_DEFENSES),
            attacks=['core_reasoning_opt5','core_reasoning'],reuse_from='hk_qwen_full',
            reuse=['query cohort','target item','order and slot','same attacked texts','extracted claims','evidence traces','CGCR renderings'],
            regenerate_attacks=False,reoptimize_attacks=False,commit='Study: Home and Kitchen Llama transfer'),
        dict(id='responsiveness',categories=list(CATEGORIES),model='qwen_4bit',queries_per_category=147,
            selection='all eligible queries among each category’s fixed 147-query cohort; report shortfall, never pad or select by rank movement',
            methods=['cgcr','details_only','seller_text'],conditions=['clean','improved'],orders=1,
            improvement='one absent, genuine-review-supported USE_CASE; calibrated verifier, same protocol as T8',
            reuse_clean_rankings=True,commit='Study: test responsiveness')]
    cfg['execution_order']=[s['id'] for s in cfg['execution_plan']]
    cfg['stage_completion']=dict(checkpoint='query granularity',commit_each_stage=True,partial_results_usable=True,
        required_artifacts=['resolved config','seed and input hashes','per-query outputs','metrics.json','run.log','local commit'])
    cfg['precision']['interpretation']='147 accepted paired cases target per standard cell; attained CIs must be reported. Adaptive 48-query subset is exploratory and has no 0.08 half-width guarantee. Responsiveness reports its eligible denominator.'
    cfg['preprocessing_reuse']=dict(extraction_key=['item_id','text_sha256','extractor_version'],
        one_clean_sanitization_per_query_candidate=True,one_attacked_target_sanitization_per_condition=True,
        details_only_attack_rankings='reuse clean result when the complete ranker input and seed are byte-identical',
        transfer='reuse Qwen-stage texts/extraction/verification, but perform Llama ranking separately',
        responsiveness='reuse main clean ranking; unchanged details-only improved input reuses clean')
    cfg['test_access_policy']='Only scripts/run_final.py after explicit approval; planning reads dev logs and configuration only'
    return cfg


def project(cfg,rates):
    """Conservative serial compute, with only guaranteed reuse credited.

    Per-category and Llama rates are explicit proxies from Home_and_Kitchen dev.
    No speculative CGCR-invariance, item overlap, or NLI-cache hit rate is assumed.
    Responsiveness budgets all 147 possible eligible improvements per category.
    """
    if cfg['study_started'] or cfg['approval']['approved']:raise ValueError('This projection requires an unstarted, unapproved plan')
    cells=[]
    def add(stage,phase,units,rate_key,*,logical_units=None,**axes):
        if units<0:raise ValueError('Negative projected work')
        rate=rates[rate_key]
        cells.append(dict(stage=stage,phase=phase,units=units,logical_units=units if logical_units is None else logical_units,
            rate_key=rate_key,seconds_per_unit=rate['seconds_per_unit'],hours=units*rate['seconds_per_unit']/3600,
            measurement=rate['source'],limitations=rate.get('limitations',[]),**axes))
    stage_grids=cfg['execution_plan'][:4]
    # All Llama preprocessing is already included in its paired Qwen H&K stage.
    for s in stage_grids:
        sid=s['id'];q=s['queries'];attacks=s['attacks'];model=s['model'];order=s['orders']
        for method in s['defenses']:
            key=model+'/'+method
            clean_key='selection_rank' if model=='qwen_4bit' and method=='no_defense' else key
            add(sid,'clean_ranking',q*order,clean_key,model=model,defense=method,queries=q,conditions=1,orders=order,
                note='Qwen no-defense clean reuses accepted target-screening result; charged once at measured screening-call cost' if clean_key=='selection_rank' else '')
            calls=q*len(attacks)*order
            add(sid,'attacked_ranking',0 if method=='details_only' else calls,key,logical_units=calls,model=model,defense=method,
                queries=q,conditions=len(attacks),orders=order,note='details-only seller-edit input is unchanged; retain every logical evaluation and reuse clean output' if method=='details_only' else '')
        if model!='qwen_4bit':continue
        screen=cfg['query_counts'][s['category']]['initial_screening_cap']
        extra=max(0,screen*rates['selection_trials_per_screened_query']['value']-q)
        add(sid,'target_screening_extra',extra,'selection_rank',screened_queries=screen,
            note='Only unsuccessful/additional screening calls here; accepted clean calls already charged in clean_ranking')
        docs=q*(10+len(attacks))
        add(sid,'extraction',docs,'extraction',queries=q,texts_per_query=10+len(attacks),
            note='New test texts only; 10 clean candidates + winning attacked target per standard attack; no item-sharing credit')
        add(sid,'verification',docs,'verification',queries=q,texts_per_query=10+len(attacks),
            note='Typed catalog/BM25/I/O overhead plus 20 NLI pairs per observed USE_CASE, no NLI-cache credit')
        add(sid,'review_indexing',q*10,'review_index',note='At most 10 new clean items/query; indexes shared across conditions and transfer')
        if 'sanitizer' in s['defenses']:add(sid,'sanitization',docs,'sanitizer_document',queries=q,documents_per_query=10+len(attacks),
            note='Each clean document once/query; only target rewritten per attack; official prompt and frozen output cap')
        if 'ppl_filter' in s['defenses']:add(sid,'perplexity_scoring',docs,'ppl_document',note='Frozen T7 threshold; no threshold fitting on test')
        if 'geo_filter' in s['defenses']:add(sid,'shield_scoring',docs,'shield_document',note='Shield filter preprocessing, then Qwen ranking; separate from full GEO-Defender')
        for attack in attacks:
            variants=5 if attack.endswith('_opt5') else 1
            add(sid,'attack_generation',q*variants,'generation/'+attack,attack=attack,variants=variants,queries=q,
                note='Generated once with Qwen, shared across defenses and Llama transfer')
            if variants==5:add(sid,'attacker_optimization_ranking',q*variants,'search_rank',attack=attack,variants=variants,queries=q)
        adaptive=s.get('adaptive')
        if adaptive:
            aq=adaptive['queries']
            add(sid,'adaptive_generation',aq*5,'adaptive_candidate',kind='unsupported USE_CASE candidate',queries=aq,variants=5)
            add(sid,'adaptive_generation',aq*2,'adaptive_reviews',kind='10-review generation chunk',queries=aq,chunks=2)
            add(sid,'adaptive_generation',aq,'adaptive_mimicry',queries=aq)
            add(sid,'extraction',aq*2,'extraction',note='One flood target text + one mimicry target text/query, shared across review budgets')
            # Flooding: B=0 flags are the same corpus. Six changed corpora plus
            # B=0 = seven verifications; mimicry is one more. Direct claim probes
            # and the five candidate probes are additional one-claim checks.
            add(sid,'verification',aq*8,'verification',note='7 flood corpora + mimicry per query, no additional clean-item extraction')
            add(sid,'adaptive_verifier_screening',aq*(5+7)*20,'nli_pair',note='At most 5 candidate probes + 7 direct requested-claim probes; 20 review sentences each')
            add(sid,'adaptive_ranking',aq*7,'adaptive_cgcr_rank',logical_units=aq*8,attack='review_flooding',defense='cgcr',queries=aq,
                note='8 logical B/flag conditions; identical B=0 verified/unverified input reused; main clean already available')
            for method in adaptive['mimicry_defenses']:
                add(sid,'adaptive_ranking',0 if method=='details_only' else aq,model+'/'+method,logical_units=aq,attack='verifiability_mimicry',defense=method,queries=aq)
            add(sid,'sanitization',aq,'sanitizer_document',note='Mimicry target only; all 9 competitors and clean target reused')
            add(sid,'perplexity_scoring',aq,'ppl_document',note='Mimicry target only')
            add(sid,'shield_scoring',aq,'shield_document',note='Mimicry target only')
    response=cfg['execution_plan'][4];rq=response['queries_per_category']
    for category in response['categories']:
        sid=response['id'];add(sid,'responsiveness_screening',rq*24*20,'nli_pair',category=category,
            note='Conservative maximum: 24 candidate genuine-use phrases/query × 20 review sentences; no cached proposal credit')
        add(sid,'extraction',rq,'response_extraction',category=category,note='At most one eligible improved target/query; final extractor, original texts unchanged')
        add(sid,'verification',rq,'verification',category=category,note='Only changed target; unchanged competitors reused')
        for method in response['methods']:
            add(sid,'responsiveness_ranking',0 if method=='details_only' else rq,'response_rank/'+method,
                logical_units=2*rq,category=category,method=method,
                note='Main clean reused; identical details-only improved input reused. Other improved inputs budget a fresh call.')
    by_stage=defaultdict(float);by_phase=defaultdict(float)
    for cell in cells:by_stage[cell['stage']]+=cell['hours'];by_phase[cell['phase']]+=cell['hours']
    return dict(formula='T_hours = sum_cells(Q × m × conditions × measured seconds/call) / 3600; preprocessing uses the same units × measured seconds/unit formula',
        total_hours=sum(c['hours'] for c in cells),total_continuous_days=sum(c['hours'] for c in cells)/24,
        stage_hours={s:by_stage[s] for s in cfg['execution_order']},phase_hours=dict(by_phase),cells=cells,rates=rates,
        responsiveness_budget=dict(queries_per_category=rq,categories=len(response['categories']),
            interpretation='Upper workload at 147 eligible improvements/category; actual eligible denominator may be smaller; report shortfall'),
        assumptions=['Serial CPU/GPU sum, with no concurrency discount; not a promised wall-clock completion time.',
            'Home_and_Kitchen timings transferred to Electronics/Tools; no test contents inspected.',
            'Llama defense rates use measured Llama/Qwen baseline ratio, not new Llama defense benchmarks.',
            'Only guaranteed reuse is credited; additional CGCR invariance, item overlap and NLI-cache hits can lower compute.',
            'No test extraction/verification/generation is treated as already cached from dev.',
            'Adaptive attempt cost includes pilot retries; pilot mimicry had 14/48 generation failures. Keep 48 nominated queries, log failures, no padding.',
            'Excludes model cold starts, human approval/annotation, idle time, resource-guard pauses and unobserved new-category transfer overhead.'])
