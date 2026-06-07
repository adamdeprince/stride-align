"""Monge-Elkan multi-token hybrid similarity — Phase D.6.

The classic record-linkage formula:

    ME(s1, s2) = (1 / |T_a|) · Σ_{t ∈ T_a} max_{u ∈ T_b} sim(t, u)

where ``T_a`` / ``T_b`` are whitespace tokenisations and ``sim`` is a
configurable per-token similarity. Asymmetric by definition. Pure
Python on top of stride-align's existing Jaro / Jaro-Winkler /
Levenshtein-ratio / Indel-ratio primitives.
"""

from __future__ import annotations

import pytest

import stride_align as sa


# ---- Identity and edge cases -------------------------------------

def test_identity() -> None:
    assert sa.monge_elkan("paul johnson", "paul johnson") == 1.0
    assert sa.monge_elkan("hello", "hello") == 1.0


def test_both_empty_returns_one() -> None:
    assert sa.monge_elkan("", "") == 1.0
    assert sa.monge_elkan("   ", "   ") == 1.0


def test_one_empty_returns_zero() -> None:
    assert sa.monge_elkan("abc", "") == 0.0
    assert sa.monge_elkan("", "abc") == 0.0
    assert sa.monge_elkan("foo bar", "   ") == 0.0


# ---- Token reordering: every s1 token finds an exact match -------

def test_token_reordering_scores_one() -> None:
    # Each token in s1 has an identical token somewhere in s2 → all
    # per-token max sims are 1.0 → average is 1.0. (Symmetric in this
    # case because every token in both sides has a perfect partner.)
    assert sa.monge_elkan("foo bar", "bar foo") == 1.0
    assert sa.monge_elkan("the quick brown fox", "fox brown quick the") == 1.0


# ---- Asymmetry: the defining property of Monge-Elkan -------------

def test_asymmetric_when_token_counts_differ() -> None:
    # 'paul' as the s1 has one token; its single max sim against
    # 'paul johnson' is 1.0 → ME = 1.0.
    # 'paul johnson' as s1 has two tokens. 'paul' has jaro=1 with the
    # one s2 token 'paul'; 'johnson' has jaro=0 with 'paul'.
    # ME = (1.0 + 0.0) / 2 = 0.5.
    assert sa.monge_elkan("paul",         "paul johnson") == 1.0
    assert sa.monge_elkan("paul johnson", "paul")         == 0.5


def test_symmetric_kwarg_averages_directions() -> None:
    fwd = sa.monge_elkan("paul", "paul johnson")
    bwd = sa.monge_elkan("paul johnson", "paul")
    sym = sa.monge_elkan("paul", "paul johnson", symmetric=True)
    assert sym == pytest.approx(0.5 * (fwd + bwd), abs=1e-12)
    # Symmetric variant is order-independent.
    assert sa.monge_elkan("paul", "paul johnson", symmetric=True) == pytest.approx(
        sa.monge_elkan("paul johnson", "paul", symmetric=True), abs=1e-12
    )


# ---- Inner-similarity selection -----------------------------------

def test_inner_jaro_default() -> None:
    # Default inner='jaro' — pin against per-token Jaro.
    # Tokens: ['hello', 'world'] vs ['hallo', 'world']
    # max-jaro(hello, ·) = max(jaro(hello,hallo), jaro(hello,world))
    # max-jaro(world, ·) = max(jaro(world,hallo), jaro(world,world)) = 1.0
    expected = (
        max(sa.jaro_similarity("hello", "hallo"), sa.jaro_similarity("hello", "world"))
        + 1.0
    ) / 2
    assert sa.monge_elkan("hello world", "hallo world") == pytest.approx(expected, abs=1e-12)


def test_inner_jaro_winkler_distinct_from_jaro() -> None:
    # Jaro-Winkler boosts common-prefix matches: same inputs as above,
    # different score because Jaro-Winkler ≠ Jaro.
    inner_j  = sa.monge_elkan("hello world", "hallo world", inner="jaro")
    inner_jw = sa.monge_elkan("hello world", "hallo world", inner="jaro_winkler")
    assert inner_jw >= inner_j
    # Both 'hello' and 'hallo' share prefix 'h' → JW boosts. Sanity-
    # check by recomputing the formula.
    expected = (
        max(sa.jaro_winkler_similarity("hello", "hallo"),
            sa.jaro_winkler_similarity("hello", "world"))
        + 1.0
    ) / 2
    assert inner_jw == pytest.approx(expected, abs=1e-12)


def test_inner_levenshtein_ratio() -> None:
    expected = (
        max(sa.levenshtein_normalized_score("hello", "hallo"),
            sa.levenshtein_normalized_score("hello", "world"))
        + 1.0
    ) / 2
    assert sa.monge_elkan("hello world", "hallo world",
                          inner="levenshtein_ratio") == pytest.approx(expected, abs=1e-12)


def test_inner_indel_ratio() -> None:
    expected = (
        max(sa.indel_normalized_score("hello", "hallo"),
            sa.indel_normalized_score("hello", "world"))
        + 1.0
    ) / 2
    assert sa.monge_elkan("hello world", "hallo world",
                          inner="indel_ratio") == pytest.approx(expected, abs=1e-12)


def test_inner_callable() -> None:
    # Custom inner: exact-match indicator. ME counts the fraction of
    # s1 tokens that appear verbatim in s2.
    score = sa.monge_elkan("a b c", "a c d",
                           inner=lambda x, y: 1.0 if x == y else 0.0)
    # 'a' and 'c' are in s2; 'b' is not → 2/3.
    assert score == pytest.approx(2 / 3, abs=1e-12)


def test_inner_unknown_string_raises() -> None:
    with pytest.raises(ValueError, match="unknown inner similarity"):
        sa.monge_elkan("a", "b", inner="not-a-real-metric")


def test_inner_wrong_type_raises() -> None:
    with pytest.raises(TypeError, match="inner must be a callable"):
        sa.monge_elkan("a", "b", inner=42)


# ---- Processor and bytes input -----------------------------------

def test_processor_preprocesses_inputs() -> None:
    # Case-insensitive matching: pass str.lower as the preprocessor.
    assert sa.monge_elkan("PAUL", "paul",            processor=str.lower) == 1.0
    assert sa.monge_elkan("PAUL JOHNSON", "paul JOHNSON", processor=str.lower) == 1.0


def test_bytes_input_widens_as_latin1() -> None:
    # Same convention as the C++ engines.
    assert sa.monge_elkan(b"paul", b"paul") == 1.0
    assert sa.monge_elkan(b"foo bar", b"bar foo") == 1.0


def test_non_str_non_bytes_raises() -> None:
    with pytest.raises(TypeError, match="must be str or bytes"):
        sa.monge_elkan(42, "hello")


# ---- Range check -------------------------------------------------

def test_result_in_unit_interval() -> None:
    import random
    random.seed(99)
    alphabet = "abcdef ghi jklmn"
    for _ in range(50):
        m, n = random.randrange(0, 30), random.randrange(0, 30)
        a = "".join(random.choice(alphabet) for _ in range(m)).strip()
        b = "".join(random.choice(alphabet) for _ in range(n)).strip()
        for inner in ["jaro", "jaro_winkler", "levenshtein_ratio", "indel_ratio"]:
            r = sa.monge_elkan(a, b, inner=inner)
            assert 0.0 <= r <= 1.0, (a, b, inner, r)


# ---- Pinned numerical fixtures (Monge & Elkan 1996 spirit) ------

# The original Monge-Elkan paper averages per-s1-token best matches.
# Pin a small fixture that exercises the partial-credit semantics.
def test_paper_style_partial_match() -> None:
    # 'paul johnson' vs 'paul jones': two s1 tokens, two s2 tokens.
    # max-jaro('paul', ·) = jaro(paul, paul) = 1.0
    # max-jaro('johnson', ·) = max(jaro(johnson, paul), jaro(johnson, jones))
    #                       = jaro(johnson, jones)
    j_jj = sa.jaro_similarity("johnson", "jones")
    expected = (1.0 + max(sa.jaro_similarity("johnson", "paul"), j_jj)) / 2
    assert sa.monge_elkan("paul johnson", "paul jones") == pytest.approx(expected, abs=1e-12)


def test_paper_style_repeated_best_match() -> None:
    # Multiple s1 tokens can pick the same s2 token as their best
    # match — the metric does not enforce a matching.
    # 'a a a' vs 'a': each s1 'a' matches s2 'a' perfectly → 1.0.
    assert sa.monge_elkan("a a a", "a") == 1.0
    # Reverse direction has only one s1 token, also perfect.
    assert sa.monge_elkan("a", "a a a") == 1.0
