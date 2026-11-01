"""FramingBench §4.4 official blind sanitizer prompt, adapted runner.

Source m1k2zoo/framingbench, commit 8a53ba8f99853fb3220b419f7ad107dea69a814b,
MIT. Same system prompt, temperature 0 and 4096 output tokens. Qwen2.5-7B 4-bit
replaces upstream Llama-3.1-70B/vLLM. Numeric candidate headers replace names;
no hidden target identity, query, attack label or catalog evidence is supplied.
Malformed/missing sections use the original whole document, explicitly flagged.
"""
import re


def comparison_document(docs):
    return '\n\n'.join(f'## Candidate {i+1}\n{text}' for i,text in enumerate(docs))


def parse_sanitized(text, n):
    sections=list(re.finditer(r'^## Candidate (\d+)\s*$',text,re.M))
    if [int(s.group(1)) for s in sections]!=list(range(1,n+1)):
        return None
    if text[:sections[0].start()].strip():return None
    docs=[text[s.end():sections[i+1].start() if i+1<n else len(text)].strip()
          for i,s in enumerate(sections)]
    return docs if all(docs) else None
