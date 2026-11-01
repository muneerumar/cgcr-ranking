"""mine category schemas from structured details only, with editable synonyms."""

from collections import Counter, defaultdict
import json
import re
import unicodedata

from cgcr.data.units import parse_attribute, observed_units

# Strip only known unit annotations, preserving semantic parentheses such as Description.
UNIT_WORDS = r'(?:kg|g|mg|lbs?|pounds?|ounces?|oz|ml|liters?|litres?|l|cm|mm|inches|inch|in|feet|ft|watts?|w|kw|volts?|v|°?[cf]|degrees? [cf])'


def normalize_key(key):
    """Normalize case/spacing/punctuation and trailing unit annotations."""
    text = unicodedata.normalize('NFKC', str(key)).casefold()
    text = ''.join(c for c in text if unicodedata.category(c) != 'Cf')
    text = re.sub(r'\s+', ' ', text).strip(' :?')
    text = re.sub(rf'\s*[\[(]\s*{UNIT_WORDS}\s*[\])]$', '', text)
    text = re.sub(rf'\s+{UNIT_WORDS}$', '', text)
    return re.sub(r'\s+', ' ', text).strip(' :?')


class KeyPolicy:
    """Resolve synonyms and non-product exclusions from the category YAML."""

    def __init__(self, config):
        self.aliases = {}
        for canonical, aliases in config.get('synonyms', {}).items():
            for alias in [canonical, *aliases]:
                key = normalize_key(alias)
                if key in self.aliases and self.aliases[key] != normalize_key(canonical):
                    raise ValueError(f'Conflicting synonym: {alias}')
                self.aliases[key] = normalize_key(canonical)
        self.excluded = {normalize_key(k): reason for k, reason in config.get('exclude', {}).items()}

    def canonical(self, key):
        normalized = normalize_key(key)
        if not normalized or normalized in self.excluded:
            return None
        return self.aliases.get(normalized, normalized)


def details_dict(value):
    """Accept the  lossless JSON representation or an already decoded mapping."""
    decoded = json.loads(value) if isinstance(value, str) else value
    if not isinstance(decoded, dict):
        raise ValueError('Catalog details must be a JSON object')
    return decoded


def has_value(value):
    return value is not None and value != '' and value != [] and value != {}


_VARIANT = re.compile(r'\b(size|colou?r|capacity|volume|pack|package quantity|count|dimensions|weight|length|width|height|diameter|depth|thickness)\b')


def is_variant_sensitive(attribute):
    """Flag observed size/color/capacity/pack/measurement keys conservatively."""
    return bool(_VARIANT.search(attribute)) or attribute.startswith('number of ')


def variant_sensitive_keys(details, policy):
    return sorted({canonical for key, value in details_dict(details).items()
                   if has_value(value) and (canonical := policy.canonical(key))
                   and is_variant_sensitive(canonical)})


def normalized_details(details, policy):
    """Keep multiple alias values for one item without inflating item coverage."""
    values = defaultdict(set)
    for key, value in details_dict(details).items():
        canonical = policy.canonical(key)
        if canonical and has_value(value):
            rendered = value.strip() if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)
            if rendered:
                values[canonical].add(rendered)
    return dict(values)


def infer_value_type(attribute, values):
    """Deterministic descriptive heuristic, exposed with parse rates for review."""
    total = sum(values.values())
    booleans = {'yes', 'no', 'true', 'false'}
    if sum(n for v, n in values.items() if v.strip().casefold() in booleans) / total >= .95:
        return 'boolean', [], 0.0
    parsed = [(parse_attribute(value, attribute), count) for value, count in values.items()]
    parsed_count = sum(n for m, n in parsed if m is not None)
    units = sorted({m.source_unit for m, _ in parsed if m is not None} | {u for value in values for u in observed_units(value)})
    if parsed_count / total >= .5:
        return 'numeric+unit', units, 100 * parsed_count / total
    avg_length = sum(len(v) * n for v, n in values.items()) / total
    kind = 'categorical' if avg_length <= 80 and (len(values) <= 100 or len(values) / total <= .2) else 'text'
    return kind, units, 100 * parsed_count / total


def build_schema(items, synonyms, category, min_coverage=.02):
    """Count each unique parent once; only details can create attributes/values."""
    if not 0 <= min_coverage <= 1:
        raise ValueError('min_coverage must lie in [0,1]')
    policy = KeyPolicy(synonyms)
    counts, aliases, values, dropped = Counter(), defaultdict(set), defaultdict(Counter), Counter()
    for item in items.values():
        details = details_dict(item['details'])
        normalized = normalized_details(details, policy)
        counts.update(normalized.keys())
        for attr, item_values in normalized.items():
            values[attr].update(item_values)
        for key, value in details.items():
            if not has_value(value):
                continue
            canonical = policy.canonical(key)
            if canonical:
                aliases[canonical].add(key)
            else:
                dropped[key] += 1
    attributes = []
    for attr, count in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0])):
        if count / len(items) < min_coverage:
            continue
        kind, units, parsed_pct = infer_value_type(attr, values[attr])
        override = synonyms.get('value_types', {}).get(attr)
        if override is not None:
            if override not in {'numeric+unit', 'categorical', 'boolean', 'text'}:
                raise ValueError(f'Invalid value type: {override}')
            kind = override
        if kind != 'numeric+unit':
            units = []
        top = sorted(values[attr].items(), key=lambda pair: (-pair[1], pair[0]))[:10]
        attributes.append({'canonical_name': attr, 'aliases': sorted(aliases[attr]),
                           'item_count': count, 'coverage_pct': 100 * count / len(items),
                           'value_type': kind, 'value_type_source': 'configured' if override else 'heuristic',
                           'observed_units': units,
                           'numeric_parse_pct': parsed_pct,
                           'top_10_example_values': [{'value': v, 'item_count': n} for v, n in top],
                           'variant_sensitive': is_variant_sensitive(attr)})
    return {'category': category, 'source': 'catalog.details only', 'scope': 'unique T2 selected parent items',
            'item_count': len(items), 'min_coverage_pct': 100 * min_coverage,
            'coverage_denominator': 'all unique items, including missing details',
            'value_type_rule': 'YAML overrides; else >=95% boolean; else >=50% parsed numeric; else short repeated values categorical; else text',
            'attributes': attributes,
            'excluded_observed_keys': dict(sorted(dropped.items())),
            'below_threshold': [{'canonical_name': attr, 'item_count': n} for attr, n in sorted(counts.items())
                                if n / len(items) < min_coverage]}
