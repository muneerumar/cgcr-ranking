"""paired masks, alias hiding and content-only improvements."""
from cgcr.defenses.cgcr.stress import mask_details, review_use_case_phrases, append_improvement


def test_masks_are_seeded_nested_and_remove_alias_leakage_for_all_candidates():
    schema={'attributes':[dict(canonical_name=k,aliases=[k.upper()],value_type='categorical',variant_sensitive=False)
                         for k in ('color','brand','material','shape')]}
    row=dict(item_id='P',product_id='P',catalog=dict(details={'COLOR':'Red','color':'Red','BRAND':'B','material':'Steel','shape':'Round'},price=10))
    zero,keys=mask_details(row,schema,0,13,1)
    two,k2=mask_details(row,schema,2,13,1);four,k4=mask_details(row,schema,4,13,1)
    assert keys==[] and zero['catalog']['details']=={}
    assert len(k2)==2 and set(k2)<=set(k4) and len(k4)==4
    assert two==mask_details(row,schema,2,13,1)[0]
    assert zero['catalog']['price']==10 and row['catalog']['details']['COLOR']=='Red'


def test_review_proposals_and_improvement_do_not_change_original_text():
    assert review_use_case_phrases(['I use it for storing rice.'])==['Suitable for storing rice.']
    assert append_improvement('Clean text','Suitable for storing rice.')=='Clean text\nSuitable for storing rice.'
