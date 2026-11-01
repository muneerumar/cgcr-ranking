"""fail-closed evidence whitelist and exact seller/catalog overlap audit."""

from collections import defaultdict

from cgcr.data.schema_builder import details_dict, variant_sensitive_keys

ATTACKABLE = {
    'esci': frozenset({'product_title', 'product_bullet_point', 'product_description', 'seller_text'}),
    'catalog': frozenset({'title', 'features', 'description'}),
}
EVIDENCE = {
    'catalog': frozenset({'details', 'price', 'average_rating', 'rating_number', 'categories'}),
    'reviews': frozenset({'rating', 'title', 'text', 'timestamp', 'verified_purchase', 'helpful_vote', 'user_id'}),
}


def classify_field(source, field):
    """Source-aware classification; unknown fields are excluded, never trusted."""
    if field in ATTACKABLE.get(source, ()):
        return 'ATTACKABLE'
    if field in EVIDENCE.get(source, ()):
        return 'EVIDENCE'
    return 'EXCLUDED'


def catalog_evidence(catalog, schema=None):
    """Select only the explicit structured catalog whitelist, without seller prose."""
    selected = {key: catalog.get(key) for key in sorted(EVIDENCE['catalog'])}
    import json
    selected['details'] = json.dumps(evidence_details(catalog, schema), sort_keys=True, ensure_ascii=False)
    return selected


def evidence_record(row, policy, schema=None):
    """Keep query/child provenance: a parent may have different children per query."""
    return {'query_id': row['query_id'], 'item_id': row['item_id'],
            'product_id': row['product_id'],
            'joined_via_variant': row['product_id'] != row['item_id'],
            'variant_sensitive_keys': variant_sensitive_keys(row['catalog']['details'], policy),
            'catalog': catalog_evidence(row['catalog'], schema), 'reviews': row['reviews']}


def _leaves(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from _leaves(child)
    elif isinstance(value, list):
        for child in value:
            yield from _leaves(child)
    elif value is not None and str(value).strip():
        yield str(value)


def overlap_audit(rows):
    """Item-weighted, case-sensitive literal substring audit across its selected texts.

    Feature overlap means at least one entire nonempty feature bullet. Details are
    nonempty scalar leaves rendered with str(), not keys or serialized containers.
    A secondary >=4 character measure exposes incidental short-value matches.
    """
    texts = defaultdict(set)
    catalogs = {}
    for row in rows:
        item = row['item_id']
        texts[item].add(row['seller_text'])
        if item in catalogs and catalogs[item] != row['catalog']:
            raise ValueError(f'Conflicting parent catalog: {item}')
        catalogs[item] = row['catalog']
    records = []
    for item, catalog in sorted(catalogs.items()):
        features = [str(v) for v in catalog.get('features', []) if str(v).strip()]
        details = list(_leaves(details_dict(catalog['details'])))
        matches = lambda v: any(v in text for text in texts[item])
        records.append({'item_id': item, 'has_features': bool(features), 'has_details_values': bool(details),
                        'feature_verbatim_overlap': any(map(matches, features)),
                        'all_features_verbatim_overlap': bool(features) and all(map(matches, features)),
                        'details_value_verbatim_overlap': any(map(matches, details)),
                        'details_value_ge4chars_verbatim_overlap': any(matches(v) for v in details if len(v) >= 4)})
    denominator = len(records)
    metrics = {'unique_items': denominator, 'method': 'case-sensitive full-value substring in any selected seller_text for the same parent'}
    for key in ('has_features', 'has_details_values', 'feature_verbatim_overlap', 'all_features_verbatim_overlap',
                'details_value_verbatim_overlap', 'details_value_ge4chars_verbatim_overlap'):
        count = sum(r[key] for r in records)
        metrics[key] = {'items': count, 'denominator': denominator,
                        'pct': 100 * count / denominator if denominator else None}
    return metrics, records


ALLOWED_DETAIL_TYPES = frozenset({'numeric+unit', 'categorical', 'boolean'})


def _schema_attributes(schema):
    return {a['canonical_name']: a for a in (schema or {}).get('attributes', [])}


def classify_detail(attribute, schema):
    """unknown keys fail closed; schema text attributes are attackable."""
    entry = _schema_attributes(schema).get(attribute)
    if entry is None:
        return 'EXCLUDED'
    return 'EVIDENCE' if entry['value_type'] in ALLOWED_DETAIL_TYPES else 'ATTACKABLE'


def evidence_details(catalog, schema):
    """Resolve raw aliases and keep only schema-typed scalar detail values."""
    from cgcr.data.schema_builder import normalize_key
    attributes = _schema_attributes(schema)
    aliases = {normalize_key(alias): name for name, entry in attributes.items()
               for alias in [name, *entry['aliases']]}
    selected = defaultdict(set)
    for key, value in details_dict(catalog.get('details') or {}).items():
        name = aliases.get(normalize_key(key))
        if classify_detail(name, schema) != 'EVIDENCE':
            continue
        for scalar in value if isinstance(value, list) else [value]:
            if isinstance(scalar, (str, int, float, bool)) and str(scalar).strip():
                selected[name].add(str(scalar).strip())
    return {name: sorted(values) for name, values in sorted(selected.items())}


def detail_match_status(attribute, claim, parent_values, schema, *, joined_via_variant=False, review_supported=False):
    """policy helper for exact matches/mismatch routing; no NLI or tolerance tuning.

    review_supported is a trusted upstream verification result, never model prose.
    Ranges remain unknown even if the same range occurs in both sources.
    """
    import math
    from cgcr.data.units import parse_attribute, unit_parse_status
    entry = _schema_attributes(schema).get(attribute)
    if classify_detail(attribute, schema) != 'EVIDENCE':
        return 'excluded_attackable' if entry else 'unknown'
    parents = parent_values if isinstance(parent_values, (list, tuple)) else [parent_values]
    if unit_parse_status(claim, attribute) == 'range_unparsed' or any(unit_parse_status(v, attribute) == 'range_unparsed' for v in parents):
        return 'range_unparsed'
    if review_supported:
        return 'review_support'
    if not parents or parents == [None]:
        return 'unknown'
    normalize = lambda v: ' '.join(str(v).casefold().split())
    if entry['value_type'] == 'numeric+unit':
        parsed = parse_attribute(claim, attribute)
        candidates = [parse_attribute(v, attribute) for v in parents]
        if parsed is None or not any(candidates):
            return 'unknown'
        equal = any(v is not None and v.unit == parsed.unit and v.axes == parsed.axes
                    and len(v.values) == len(parsed.values)
                    and all(math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-9) for a,b in zip(v.values,parsed.values)) for v in candidates)
    else:
        equal = any(normalize(claim) == normalize(v) for v in parents)
    if equal:
        return 'support'
    if joined_via_variant and entry['variant_sensitive']:
        return 'unknown_variant'
    return 'contradiction'


def canonical_title(catalog, schema, licensed_variant_attributes=None):
    """structured brand + leaf category + explicitly licensed variant values.

    Catalog/seller title, features, descriptions, and ESCI brand never contribute.
    Review-supported values may differ from the parent. All others must match it.
    """
    details = evidence_details(catalog, schema)
    brand = details.get('brand', [])
    categories = catalog.get('categories') or []
    leaf = next((str(v).strip() for v in reversed(categories) if str(v).strip()), 'Uncategorized product')
    parts = ([brand[0]] if len(brand) == 1 else []) + [leaf]
    licensed = licensed_variant_attributes or {}
    for name in ('capacity', 'size', 'color', 'unit count', 'number of items', 'number of pieces', 'package quantity'):
        record = licensed.get(name)
        if not record or classify_detail(name, schema) != 'EVIDENCE':
            continue
        status = record.get('status')
        value = record.get('value')
        if value is None or status not in {'support', 'review_support'}:
            continue
        actual = detail_match_status(name, value, details.get(name, []), schema, review_supported=status == 'review_support')
        if actual in {'support', 'review_support'}:
            parts.append(f'{name}: {value}')
    return ' | '.join(parts)
