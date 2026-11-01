"""deterministic rendering from licensed assertions and platform evidence only."""
import json
from cgcr.data.evidence import canonical_title

def render(row,decisions,schema):
    order={a['canonical_name']:i for i,a in enumerate(schema['attributes'])}
    licensed=set()
    variant={}
    for d in decisions:
        if not d.get('licensed'):continue
        name=d.get('render_attribute');value=d.get('render_value')
        if not name or not isinstance(value,str) or not value.strip():raise ValueError('Licensed decision lacks a canonical rendering')
        licensed.add((name,' '.join(value.split())))
    for name,value in sorted(licensed):
        if name not in variant:variant[name]={'status':'support','value':value}
    title=canonical_title(row['catalog'],schema,variant)
    lines=[title]
    for name,value in sorted(licensed,key=lambda p:(order.get(p[0],len(order)),p[0],p[1])):
        # JSON strings delimit evidence values and normalize escaping deterministically.
        lines.append(name+': '+json.dumps(value,ensure_ascii=False))
    for name in ('price','average_rating','rating_number'):
        value=row['catalog'].get(name)
        if value is not None and str(value).strip():lines.append(name+': '+json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False))
    return '\n'.join(lines)
