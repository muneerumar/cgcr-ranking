"""streaming sentence split, metadata, BM25, and atomic reuse."""
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from cgcr.data.amazon_reviews_2023 import REVIEW_SCHEMA
from cgcr.data.review_sentences import split_sentences,build_item_index,SentenceIndex

def test_split_preserves_decimals_and_html_boundaries():
    assert split_sentences('Holds 1.42 liters. Works!<br/>Weight is 2.29 pounds.')==['Holds 1.42 liters.','Works!','Weight is 2.29 pounds.']
    assert split_sentences(None)==[]

def test_index_bm25_metadata_atomicity_and_reuse(tmp_path):
    source=tmp_path/'reviews.parquet';target=tmp_path/'P.sqlite'
    rows=[dict(rating=5.,title='Capacity',text='Holds two liters. Stays cool.',timestamp=1,verified_purchase=True,helpful_vote=3,user_id='hashed-u1'),
          dict(rating=2.,title='',text='The red color fades.',timestamp=2,verified_purchase=False,helpful_vote=0,user_id='hashed-u2')]
    pq.write_table(pa.Table.from_pylist(rows,schema=REVIEW_SCHEMA),source)
    result=build_item_index(source,target,'P')
    assert result['reviews']==2 and result['sentences']==4 and not result['reused']
    with SentenceIndex(target) as index:
        found=index.retrieve('two liters',3)
        assert found and found[0].text=='Holds two liters.'
        assert found[0].user_id=='hashed-u1' and found[0].verified_purchase and found[0].helpful_vote==3
        assert index.retrieve('" OR * --',0)==[]
        assert index.retrieve('nonexistentword')==[]
        with pytest.raises(ValueError):index.retrieve('x',-1)
    assert build_item_index(source,target,'P')['reused']
    assert not list(tmp_path.glob('*.building*')) and not list(tmp_path.glob('.*.building*'))

def test_empty_reviews_make_valid_empty_index(tmp_path):
    source=tmp_path/'empty.parquet';pq.write_table(pa.Table.from_pylist([],schema=REVIEW_SCHEMA),source)
    result=build_item_index(source,tmp_path/'empty.sqlite','E')
    assert result['sentences']==result['reviews']==0
    with SentenceIndex(tmp_path/'empty.sqlite') as index:assert index.retrieve('capacity')==[]

def test_probe_skips_emoji_only_title(tmp_path):
    from cgcr.data.review_sentences import probe_index
    source=tmp_path/'emoji.parquet'
    row=dict(rating=5.,title='👍',text='Holds two liters.',timestamp=1,verified_purchase=True,helpful_vote=0,user_id='hash')
    pq.write_table(pa.Table.from_pylist([row],schema=REVIEW_SCHEMA),source)
    target=tmp_path/'P.sqlite';build_item_index(source,target,'P')
    with SentenceIndex(target) as index:
        result=probe_index(index)
        assert result['has_searchable_sentence'] and result['retrieval_nonempty']
        assert result['skipped_nonword_sentences']==1
