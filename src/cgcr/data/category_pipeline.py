"""CPU lane 1: disk-backed  preparation using retained interim files only.

Reuse seeded  candidate selection and  schema/evidence policies, without a
whole-ESCI product table in memory or writes to Home_and_Kitchen artifacts.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import time
import pyarrow as pa
import pyarrow.parquet as pq
from cgcr.data.amazon_reviews_2023 import CATALOG_SCHEMA
from cgcr.data.candidate_sets import build_candidates,split_queries,select_pilot,LABEL_ORDER
from cgcr.data.evidence import evidence_record,overlap_audit,EVIDENCE
from cgcr.data.schema_builder import build_schema,KeyPolicy
from cgcr.utils.io import read_yaml,write_yaml,write_jsonl

CATEGORIES=('Electronics','Tools_and_Home_Improvement')

def save(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True);write_jsonl(path,[value])

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block:=stream.read(1024*1024):h.update(block)
    return h.hexdigest()

def records(path,columns=None):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=256,columns=columns,use_threads=False):
        yield from batch.to_pylist()

def candidate_schema():
    return pa.schema([(key,pa.string()) for key in ('query_id','query','item_id','esci_label','seller_text','product_id','product_brand','source_split')]
        + [('catalog',pa.struct(list(CATALOG_SCHEMA)[1:])),('reviews',pa.string()),('utility_eligible',pa.bool_())])

def connect(path):
    db=sqlite3.connect(path)
    db.execute('PRAGMA cache_size=-2048');db.execute('PRAGMA temp_store=FILE')
    db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=NORMAL')
    return db

def prepare_category(root,category,seed=13,pilot_size=300,min_utility=200,progress=lambda _:None):
    if category not in CATEGORIES:raise ValueError('CPU lane cannot modify the running Home_and_Kitchen inputs')
    root=Path(root);interim=root/'data/interim'/category;out=root/'data/processed'/category
    stage=out/'.cpu_build';stage.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();pa.set_cpu_count(4);pa.set_io_thread_count(1)
    study=json.loads((interim/'study.json').read_text());qids=study['qualifying_query_ids']
    splits=split_queries(qids,seed);dev=set(splits['dev']);wanted=set(qids)
    synonyms=read_yaml(root/'configs/schemas'/f'{category}_synonyms.yaml')
    source_paths=[interim/name for name in ('metadata.parquet','asin_parent_map.parquet','study.json')]
    source_paths += [root/'data/interim/esci'/f'{name}.parquet' for name in ('examples','products')]
    inputs={str(p.relative_to(root)):digest(p) for p in source_paths}
    dbpath=stage/'join.sqlite';dbpath.unlink(missing_ok=True)
    db=connect(dbpath)
    try:
        db.executescript('CREATE TABLE catalog(parent TEXT PRIMARY KEY,payload TEXT); CREATE TABLE examples(qid TEXT,asin TEXT,parent TEXT,payload TEXT); CREATE INDEX query_idx ON examples(qid); CREATE TABLE products(asin TEXT PRIMARY KEY,payload TEXT); CREATE TABLE candidates(item TEXT,payload TEXT); CREATE INDEX item_idx ON candidates(item);')
        parents=set()
        for row in records(interim/'metadata.parquet'):
            parent=row['parent_asin'];parents.add(parent)
            db.execute('INSERT INTO catalog VALUES (?,?)',(parent,json.dumps(row,ensure_ascii=False)))
        db.commit()
        mapping={r['asin']:r['parent_asin'] for r in records(interim/'asin_parent_map.parquet')}
        needed=set();kept_examples=0
        for row in records(root/'data/interim/esci/examples.parquet'):
            if row['query_id'] not in wanted:continue
            asin=row['product_id'];parent=asin if asin in parents else mapping.get(asin)
            if parent not in parents:continue
            needed.add(asin);kept_examples+=1
            db.execute('INSERT INTO examples VALUES (?,?,?,?)',(row['query_id'],asin,parent,json.dumps(row,ensure_ascii=False)))
        db.commit();del mapping,parents
        progress(f'{category}: {kept_examples} matching examples; {len(needed)} seller records needed')
        count=0
        for row in records(root/'data/interim/esci/products.parquet'):
            if row['product_id'] in needed:
                db.execute('INSERT OR REPLACE INTO products VALUES (?,?)',(row['product_id'],json.dumps(row,ensure_ascii=False)))
                count+=1
            if count and count%1000==0:db.commit()
        db.commit();del needed
        schema=candidate_schema();eligibility={};items={};selected_items=set();labels=Counter();per_set=Counter();query_rows=[]
        temp={name:stage/f'{name}.parquet' for name in ('candidates','dev','test')}
        writers={name:pq.ParquetWriter(path,schema) for name,path in temp.items()}
        try:
            for index,qid in enumerate(sorted(qids)):
                pool={};catalog={}
                cursor=db.execute('SELECT e.payload,p.payload,e.parent,c.payload FROM examples e JOIN products p ON p.asin=e.asin JOIN catalog c ON c.parent=e.parent WHERE e.qid=?',(qid,))
                for er,pr,parent,cr in cursor:
                    row=json.loads(er)|json.loads(pr)|dict(item_id=parent)
                    previous=pool.get(parent)
                    if previous is None or (LABEL_ORDER[row['esci_label']],row['product_id'])<(LABEL_ORDER[previous['esci_label']],previous['product_id']):pool[parent]=row
                    catalog[parent]=json.loads(cr)
                candidates=build_candidates(qid,list(pool.values()),catalog,seed,category)
                eligible=any(r['esci_label']!='E' for r in candidates);eligibility[qid]=eligible
                table=pa.Table.from_pylist(candidates,schema=schema)
                writers['candidates'].write_table(table);writers['dev' if qid in dev else 'test'].write_table(table)
                for r in candidates:
                    item=r['item_id'];selected_items.add(item);items[item]={'details':r['catalog']['details']}
                    db.execute('INSERT INTO candidates VALUES (?,?)',(item,json.dumps(r,ensure_ascii=False)))
                counts=dict(Counter(r['esci_label'] for r in candidates));labels.update(counts);per_set[json.dumps(counts,sort_keys=True)]+=1
                query_rows.append(dict(query_id=qid,split='dev' if qid in dev else 'test',utility_eligible=eligible,labels=counts))
                if index%100==0:db.commit();progress(f'{category}: candidates {index+1}/{len(qids)}')
        finally:
            for writer in writers.values():writer.close()
        db.commit()
        schema_config=build_schema(items,synonyms,category,.02);policy=KeyPolicy(synonyms)
        evidence_schema=pa.schema([(key,pa.string()) for key in ('query_id','item_id','product_id')]
            + [('joined_via_variant',pa.bool_()),('variant_sensitive_keys',pa.list_(pa.string())),
               ('catalog',pa.struct([f for f in schema.field('catalog').type if f.name in EVIDENCE['catalog']])),('reviews',pa.string())])
        pilot_ids=select_pilot(dev,eligibility,seed,pilot_size,min_utility)
        pilot_set=set(pilot_ids);pilot_items=set();pilot_labels=Counter();variant_rows=0
        with pq.ParquetWriter(stage/'evidence.parquet',evidence_schema) as evidence_writer, pq.ParquetWriter(stage/'pilot.parquet',schema) as pilot_writer:
            for batch in pq.ParquetFile(temp['candidates']).iter_batches(batch_size=256,use_threads=False):
                rows=batch.to_pylist();ev=[evidence_record(row,policy,schema_config) for row in rows]
                evidence_writer.write_table(pa.Table.from_pylist(ev,schema=evidence_schema));variant_rows+=sum(r['joined_via_variant'] for r in ev)
                chosen=[r for r in rows if r['query_id'] in pilot_set]
                if chosen:
                    pilot_writer.write_table(pa.Table.from_pylist(chosen,schema=schema));pilot_items.update(r['item_id'] for r in chosen);pilot_labels.update(r['esci_label'] for r in chosen)
        overlap_counts=Counter();item_records=[]
        for item in sorted(selected_items):
            rows=[json.loads(r[0]) for r in db.execute('SELECT payload FROM candidates WHERE item=?',(item,))]
            _,audit=overlap_audit(rows);record=audit[0];item_records.append(record)
            overlap_counts.update({k:int(v) for k,v in record.items() if isinstance(v,bool)})
        (out/'reviews').mkdir(exist_ok=True)
        for item in sorted(selected_items):
            destination=out/'reviews'/f'{item}.parquet'
            if not destination.exists():
                source=interim/'reviews'/f'{item}.parquet'
                try:destination.hardlink_to(source)
                except OSError:shutil.copy2(source,destination)
        for name in ('candidates','dev','test','pilot','evidence'):(stage/f'{name}.parquet').replace(out/f'{name}.parquet')
        save(out/'split_ids.json',dict(seed=seed,dev_fraction=.3,query_ids=splits))
        save(out/'pilot_query_ids.json',dict(category=category,seed=seed,source_split='dev',sampling='T3 seeded proportional utility strata',query_ids=pilot_ids))
        write_yaml(root/'configs/schemas'/f'{category}.yaml',schema_config)
        write_jsonl(out/'candidate_labels.jsonl',query_rows);write_jsonl(out/'overlap_audit.jsonl',item_records)
        summary=dict(category=category,seed=seed,queries=len(qids),candidate_rows=len(qids)*10,unique_items=len(selected_items),
            utility_by_split={name:dict(queries=len(ids),utility_eligible=sum(eligibility[q] for q in ids),all_exact=sum(not eligibility[q] for q in ids)) for name,ids in splits.items()},
            pilot=dict(queries=len(pilot_ids),unique_items=len(pilot_items),utility_eligible=sum(eligibility[q] for q in pilot_ids),labels=dict(pilot_labels)),
            evidence=dict(rows=len(qids)*10,joined_via_variant_rows=variant_rows),schema=dict(attributes=len(schema_config['attributes']),items=len(items),min_coverage_pct=2.),
            labels=dict(labels),per_set_label_distribution=dict(per_set),overlap={k:dict(items=n,denominator=len(items),pct=100*n/len(items)) for k,n in overlap_counts.items()},
            elapsed_s=time.monotonic()-started,input_sha256=inputs,downloads=0)
        save(out/'cpu_lane_summary.json',summary);return summary,query_rows
    finally:
        db.close()
        for suffix in ('','-wal','-shm'):Path(str(dbpath)+suffix).unlink(missing_ok=True)
