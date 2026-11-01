"""pre-attack boundary caps and conservative, auditable edit budgets."""
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class TextEdit:
    start: int
    end: int
    replacement: str


@dataclass(frozen=True)
class TextBudget:
    base_chars: int = 1000
    attack_chars: int = 400

    def clean(self, text: str) -> str:
        """Keep whole sentences/bullets; never cut a sentence to fit the cap.

        A text with no complete boundary before the cap becomes empty. This rare
        case is counted by the runner, rather than silently cutting a word.
        """
        if len(text) <= self.base_chars:
            return text
        ends = [m.end() for m in re.finditer(r'\n|[.!?](?=\s|$)', text[:self.base_chars+1])
                if m.end() <= self.base_chars]
        return text[:max(ends)].rstrip() if ends else ''

    def apply(self, clean: str, edits: list[TextEdit]):
        """Charge max(deleted, inserted) per edit, including rewritten spans.

        This is intentionally stricter than net growth. Offsets refer to the
        capped original; edits must be sorted and disjoint. No post-attack cap.
        """
        if len(clean) > self.base_chars:
            raise ValueError('Clean input must be capped before applying edits')
        previous = 0
        charged = added = removed = 0
        parts = []
        for edit in edits:
            if not 0 <= previous <= edit.start <= edit.end <= len(clean):
                raise ValueError('Invalid, overlapping or unordered edit offsets')
            if not isinstance(edit.replacement, str):
                raise TypeError('Replacement must be text')
            parts += [clean[previous:edit.start], edit.replacement]
            previous = edit.end
            charged += max(edit.end-edit.start, len(edit.replacement))
            added += len(edit.replacement)
            removed += edit.end-edit.start
        parts.append(clean[previous:])
        attacked = ''.join(parts)
        if charged > self.attack_chars or len(attacked) > self.base_chars+self.attack_chars:
            raise ValueError('Attack exceeds character allowance; reject, never truncate')
        return attacked, {'budget_chars_used': charged, 'chars_inserted': added,
                          'chars_deleted': removed, 'clean_chars': len(clean),
                          'attacked_chars': len(attacked)}
