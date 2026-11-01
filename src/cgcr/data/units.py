"""conservative numeric/unit parsing for catalog schema inspection (no verification)."""

from dataclasses import dataclass
import re
import unicodedata


@dataclass(frozen=True)
class Measurement:
    """Canonical values and source axis order; unlabelled axes stay unlabelled."""

    values: tuple[float, ...]
    unit: str
    dimension: str
    source_unit: str
    axes: tuple[str, ...] = ()


# Exact definitions: US liquid measures; mass ounces never imply fluid ounces.
_UNITS = {
    'weight': ('g', {
        'g': 1, 'gram': 1, 'grams': 1, 'kg': 1000, 'kilogram': 1000,
        'kilograms': 1000, 'mg': .001, 'milligrams': .001,
        'lb': 453.59237, 'lbs': 453.59237, 'pound': 453.59237, 'pounds': 453.59237,
        'oz': 28.349523125, 'ounce': 28.349523125, 'ounces': 28.349523125,
    }),
    'volume': ('ml', {
        'ml': 1, 'milliliter': 1, 'milliliters': 1, 'millilitre': 1, 'millilitres': 1,
        'l': 1000, 'liter': 1000, 'liters': 1000, 'litre': 1000, 'litres': 1000,
        'fluid ounce': 29.5735295625, 'fluid ounces': 29.5735295625,
        'fl oz': 29.5735295625, 'fl. oz.': 29.5735295625,
        'cup': 236.5882365, 'cups': 236.5882365,
        'pint': 473.176473, 'pints': 473.176473,
        'quart': 946.352946, 'quarts': 946.352946,
        'gallon': 3785.411784, 'gallons': 3785.411784,
    }),
    'dimensions': ('cm', {
        'cm': 1, 'centimeter': 1, 'centimeters': 1, 'centimetres': 1,
        'mm': .1, 'millimeter': .1, 'millimeters': .1,
        'm': 100, 'meter': 100, 'meters': 100,
        'in': 2.54, 'inch': 2.54, 'inches': 2.54, '"': 2.54,
        'ft': 30.48, 'foot': 30.48, 'feet': 30.48, "'": 30.48,
    }),
    'wattage': ('W', {'w': 1, 'watt': 1, 'watts': 1, 'kw': 1000, 'kilowatts': 1000}),
    'voltage': ('V', {'v': 1, 'volt': 1, 'volts': 1}),
    'count': ('count', {'': 1, 'count': 1, 'counts': 1, 'piece': 1, 'pieces': 1,
        'pack': 1, 'packs': 1, 'item': 1, 'items': 1, 'pc': 1, 'pcs': 1}),
}
NUMBER = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)'
_COUNT_NUMBER = r'[+-]?(?:\d+(?:\.\d+)?|\.\d+)'
_COUNT_RANGE = rf'{_COUNT_NUMBER}(?:\s*(?:-|–|—|~|to)\s*{_COUNT_NUMBER})?'
COUNT_MENTIONS = re.compile(
    rf'(?<![\w.])(?:(?:set|pack|count)\s+of\s+{_COUNT_RANGE}|'
    rf'{_COUNT_RANGE}\s*-?\s*(?:pieces?|items?|counts?|packs?|pcs?))'
    r'(?!\w|\.\d|\s*(?:-|–|—|~|to)\s*\d)', re.I)


def _clean(value):
    text = unicodedata.normalize('NFKC', str(value))
    text = text.translate(str.maketrans({'″': '"', '”': '"', '′': "'", '×': 'x'}))
    text = re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))', '', text)
    return re.sub(r'\s+', ' ', text).strip().lower()


def parse_quantity(value, dimension):
    """Parse a complete scalar with explicit units; do not guess ambiguous units."""
    if dimension not in _UNITS:
        raise ValueError(f'Unsupported dimension: {dimension}')
    text = _clean(value)
    if dimension == 'count':
        # Lexical packaging counts have no mass/volume conversion. Ranges and
        # ambiguous compound counts deliberately remain unknown.
        prefix = re.fullmatch(rf'(?:set|pack|count)\s+of\s+({NUMBER})', text)
        text = (prefix[1] + ' count') if prefix else re.sub(r'(?<=\d)\s*-\s*(?=[a-z])', ' ', text)
    match = re.fullmatch(rf'({NUMBER})\s*(.*?)', text)
    if not match:
        return None
    number, source = float(match[1]), match[2]
    unit, conversions = _UNITS[dimension]
    if source not in conversions or number < 0 or (dimension == 'count' and not number.is_integer()):
        return None
    return Measurement((number * conversions[source],), unit, dimension, source or 'count')


def parse_weight(value):
    return parse_quantity(value, 'weight')


def parse_volume(value):
    return parse_quantity(value, 'volume')


def parse_wattage(value):
    return parse_quantity(value, 'wattage')


def parse_dimensions(value):
    """Parse 1–3 dimensions, propagating a trailing unit; preserve D/W/H/L labels."""
    parts = re.split(r'\s*x\s*', _clean(value))
    if not 1 <= len(parts) <= 3:
        return None
    components = []
    for part in parts:
        match = re.fullmatch(rf'({NUMBER})\s*("|\'|[a-z]+)?\s*([dwlh])?', part)
        if not match or float(match[1]) < 0:
            return None
        components.append((float(match[1]), match[2], (match[3] or '').upper()))
    conversions = _UNITS['dimensions'][1]
    trailing = components[-1][1]
    if not trailing or trailing not in conversions:
        return None
    labels = tuple(axis for _, _, axis in components)
    if any(labels) and (not all(labels) or len(set(labels)) != len(labels)):
        return None
    numbers, sources = [], []
    for number, unit, _ in components:
        unit = unit or trailing
        if unit not in conversions:
            return None
        numbers.append(number * conversions[unit]); sources.append(unit)
    return Measurement(tuple(numbers), 'cm', 'dimensions', ','.join(dict.fromkeys(sources)), labels)


def parse_temperature(value):
    """Parse one explicit temperature in C/F/K; ranges remain unparsed."""
    match = re.fullmatch(rf'({NUMBER})\s*(?:degrees?\s*|°\s*)?(celsius|fahrenheit|kelvin|[cfk])', _clean(value))
    if not match:
        return None
    number, source = float(match[1]), match[2]
    if source in ('f', 'fahrenheit'):
        number = (number - 32) * 5 / 9
    elif source in ('k', 'kelvin'):
        number -= 273.15
    if number < -273.15:
        return None
    return Measurement((number,), 'degC', 'temperature', source)


def parse_attribute(value, attribute):
    """Select a parser using the canonical attribute; None means unknown."""
    name = attribute.replace('_', ' ')
    if 'weight' in name or 'load' in name:
        return parse_weight(value)
    if any(word in name for word in ('volume', 'capacity')):
        return parse_volume(value)
    if any(word in name for word in ('dimensions', 'length', 'width', 'height', 'diameter', 'depth', 'thickness')):
        return parse_dimensions(value)
    if 'wattage' in name:
        return parse_wattage(value)
    if 'temperature' in name:
        return parse_temperature(value)
    if 'voltage' in name:
        return parse_quantity(value, 'voltage')
    if 'count' in name or name.startswith('number of') or name == 'package quantity':
        return parse_quantity(value, 'count')
    return None


def observed_units(value):
    """Recognize unit suffixes even when scalar parsing rejects an ambiguous/range value.

    Observed does not mean convertible: e.g. capacity ounces can mean mass or fluid.
    """
    text = _clean(value)
    tokens = {token for _, conversions in _UNITS.values() for token in conversions if token}
    tokens.update({'celsius', 'fahrenheit', 'kelvin', 'degrees celsius', 'degrees fahrenheit', 'degrees kelvin'})
    for token in sorted(tokens, key=lambda t: (-len(t), t)):
        if text.endswith(token):
            prefix = text[:-len(token)]
            if prefix and re.search(r'\d', prefix) and (prefix[-1].isspace() or prefix[-1].isdigit() or token in ('"', "'")):
                return {token}
    return set()


def unit_parse_status(value, attribute):
    """distinguish numeric ranges from scalar parsing failures."""
    text = _clean(value)
    if re.search(r'\d(?:\.\d+)?\s*(?:-|–|—|~|to)\s*[+-]?\d', text):
        return 'range_unparsed'
    return 'parsed' if parse_attribute(value, attribute) is not None else 'unknown'
