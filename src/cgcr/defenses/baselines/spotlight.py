"""reimplementation: spotlighting by explicit untrusted-data marking.

Data marking, not randomized encoding; delimiters escaped inside source text.
Candidates retain original order and are treated equally, including clean inputs.
"""
import html


def mark_document(text):
    return '<untrusted_product_data>\n' + html.escape(text, quote=False) + '\n</untrusted_product_data>'
