"""v2: finite grammar copies only source fragments and reserves EOS tokens."""
from cgcr.defenses.cgcr.mlx_copy_lines import CopyLineConstraint,source_phrases


class Tokenizer:
    eos_token_ids={0}
    def encode(self,text,**kwargs):return [ord(c) for c in text]


def test_fragments_are_exact_source_substrings_and_word_bounded():
    source='For mixing bread dough; a useful kitchen tool. One two three four five six seven eight nine ten.'
    phrases=source_phrases(source)
    assert phrases and all(p in source and len(p)<=200 for p in phrases)
    assert 'For mixing bread dough' in phrases
    assert 'One two three four five six seven eight nine ten' in phrases
    assert 'One two three four five six seven eight' not in phrases


def test_complete_line_allows_eos_and_prevents_none_after_a_claim():
    parser=CopyLineConstraint(Tokenizer(),['baking bread'])
    tokens=[999,998];assert ord('U') in parser.allowed(tokens)
    for char in 'USE_CASE | baking bread\n':
        assert ord(char) in parser.allowed(tokens);tokens.append(ord(char))
    allowed=parser.allowed(tokens)
    assert 0 in allowed and ord('N') not in allowed


def test_one_line_limit_and_none_have_only_eos_after_completion():
    for output in ('NONE','USE_CASE | baking\n'):
        parser=CopyLineConstraint(Tokenizer(),['baking'],max_lines=1)
        tokens=[999];parser.allowed(tokens)
        for char in output:
            assert ord(char) in parser.allowed(tokens);tokens.append(ord(char))
        assert parser.allowed(tokens)==[0]


def test_token_budget_excludes_unfinishable_phrases_before_they_start():
    parser=CopyLineConstraint(Tokenizer(),['a very long phrase'],max_generated=10)
    assert parser.allowed([999])==[ord('N')]


def test_same_span_cannot_be_generated_again_under_a_different_type():
    parser=CopyLineConstraint(Tokenizer(),['baking bread'])
    tokens=[999];parser.allowed(tokens)
    for char in 'USE_CASE | baking bread\n':
        assert ord(char) in parser.allowed(tokens);tokens.append(ord(char))
    assert parser.allowed(tokens)==[0]


def test_source_phrases_never_end_with_dangling_words_or_cut_long_clauses():
    text='A sentence ending in a; a useful complete phrase. '+('word '*50)+'end.'
    assert source_phrases(text)==['a useful complete phrase']
