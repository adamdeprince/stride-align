"""Character n-gram set similarity — Phase D.1.

Four metrics over multisets of character n-grams:

* ``sa.jaccard(a, b, n=2)`` — ``|A ∩ B| / |A ∪ B|``
* ``sa.dice(a, b, n=2)`` — ``2 · |A ∩ B| / (|A| + |B|)``
* ``sa.cosine(a, b, n=2)`` — ``⟨A, B⟩ / (‖A‖ · ‖B‖)`` over multiset
  frequency vectors
* ``sa.overlap(a, b, n=2)`` — ``|A ∩ B| / min(|A|, |B|)``

All four take ``n`` as a keyword-only argument (default 2). Hand-
computed expected values pin the behaviour on the textbook fixtures.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import stride_align as sa


# ---- Pinned numeric values ----------------------------------------
#
# ABCBDAB bigrams: AB, BC, CB, BD, DA, AB → multiset {AB:2, BC:1, CB:1, BD:1, DA:1}, size=6
# BDCAB   bigrams: BD, DC, CA, AB        → multiset {BD:1, DC:1, CA:1, AB:1},        size=4
# intersection = min(AB:2,AB:1) + min(BD:1,BD:1) = 1 + 1 = 2
# jaccard = 2 / (6 + 4 - 2)            = 2 / 8         = 0.25
# dice    = 2 * 2 / (6 + 4)            = 4 / 10        = 0.4
# overlap = 2 / min(6, 4)              = 2 / 4         = 0.5
# cosine:
#   dot      = 2·1 (AB) + 1·1 (BD)     = 3
#   ||A||²   = 2² + 1 + 1 + 1 + 1      = 8
#   ||B||²   = 1 + 1 + 1 + 1           = 4
#   cosine   = 3 / sqrt(8 · 4)         = 3 / sqrt(32)

@pytest.mark.parametrize(
    "fn, expected",
    [
        (sa.jaccard, 0.25),
        (sa.dice,    0.4),
        (sa.overlap, 0.5),
        (sa.cosine,  3.0 / math.sqrt(32.0)),
    ],
)
def test_canonical_abcbdab_bdcab(fn, expected) -> None:
    assert fn("ABCBDAB", "BDCAB") == pytest.approx(expected, abs=1e-12)


def test_hello_vs_help_n3() -> None:
    # 'hello' trigrams: hel, ell, llo (3)
    # 'help'  trigrams: hel, elp        (2)
    # intersect = 1, union = 4 → jaccard = 0.25
    assert sa.jaccard("hello", "help", n=3) == pytest.approx(0.25, abs=1e-12)


def test_aaaa_vs_aaaa_n2_multiset_semantics() -> None:
    # 'aaaa' bigrams: aa, aa, aa → multiset {aa: 3}, size 3
    # identical inputs → identity
    assert sa.jaccard("aaaa", "aaaa") == 1.0
    assert sa.dice("aaaa", "aaaa")    == 1.0
    assert sa.cosine("aaaa", "aaaa")  == 1.0
    assert sa.overlap("aaaa", "aaaa") == 1.0


def test_aaaa_vs_aaa_n2_partial_multiset_overlap() -> None:
    # 'aaaa' bigrams: aa, aa, aa  → multiset {aa: 3}, size 3
    # 'aaa'  bigrams: aa, aa      → multiset {aa: 2}, size 2
    # intersection = min(3, 2)    = 2
    # jaccard = 2 / (3 + 2 - 2)   = 2 / 3
    # dice    = 4 / 5             = 0.8
    # overlap = 2 / 2             = 1.0
    # cosine: dot = 3·2 = 6, ||A||² = 9, ||B||² = 4, cosine = 6/sqrt(36) = 1.0
    assert sa.jaccard("aaaa", "aaa") == pytest.approx(2 / 3, abs=1e-12)
    assert sa.dice("aaaa", "aaa")    == pytest.approx(0.8, abs=1e-12)
    assert sa.overlap("aaaa", "aaa") == 1.0
    assert sa.cosine("aaaa", "aaa")  == pytest.approx(1.0, abs=1e-12)


# ---- Identity / edge cases ----------------------------------------

@pytest.mark.parametrize("fn", [sa.jaccard, sa.dice, sa.cosine, sa.overlap])
def test_identical_inputs_score_one(fn) -> None:
    assert fn("hello", "hello") == 1.0


@pytest.mark.parametrize("fn", [sa.jaccard, sa.dice, sa.cosine, sa.overlap])
def test_both_empty_score_one(fn) -> None:
    # Convention: vacuously identical.
    assert fn("", "") == 1.0


@pytest.mark.parametrize("fn", [sa.jaccard, sa.dice, sa.cosine, sa.overlap])
def test_one_empty_scores_zero(fn) -> None:
    assert fn("abc", "") == 0.0
    assert fn("", "abc") == 0.0


@pytest.mark.parametrize("fn", [sa.jaccard, sa.dice, sa.cosine, sa.overlap])
def test_disjoint_alphabets_score_zero(fn) -> None:
    assert fn("abc", "xyz") == 0.0


@pytest.mark.parametrize("fn", [sa.jaccard, sa.dice, sa.cosine, sa.overlap])
def test_shorter_than_n_treated_as_empty(fn) -> None:
    # |input| < n → empty multiset → identity rules apply.
    # Both have empty multisets at n=5 → score 1.0.
    assert fn("ab", "cd", n=5) == 1.0
    # One has empty multiset, the other does not → score 0.0.
    assert fn("ab", "abcdef", n=5) == 0.0


# ---- Symmetry, range, monotone -----------------------------------

@pytest.mark.parametrize("fn", [sa.jaccard, sa.dice, sa.cosine, sa.overlap])
def test_symmetric(fn) -> None:
    # All four formulas are symmetric in a and b (unlike
    # Ratcliff-Obershelp).
    pairs = [
        ("ABCBDAB", "BDCAB"),
        ("hello", "world"),
        ("aaaa", "aaa"),
        ("kitten", "sitting"),
    ]
    for a, b in pairs:
        assert fn(a, b) == pytest.approx(fn(b, a), abs=1e-12)


@pytest.mark.parametrize("fn", [sa.jaccard, sa.dice, sa.cosine, sa.overlap])
def test_result_in_unit_interval(fn) -> None:
    import random
    random.seed(31)
    for _ in range(60):
        m, n = random.randrange(0, 30), random.randrange(0, 30)
        alphabet = random.choice(["abc", "abcdef"])
        a = "".join(random.choice(alphabet) for _ in range(m))
        b = "".join(random.choice(alphabet) for _ in range(n))
        r = fn(a, b)
        assert 0.0 <= r <= 1.0, (fn.__name__, a, b, r)


# ---- Dice / Jaccard closed-form relation -------------------------

def test_dice_jaccard_relation() -> None:
    # The closed-form relationship: D = 2J / (1 + J).
    pairs = [
        ("ABCBDAB", "BDCAB"),
        ("hello", "world"),
        ("kitten", "sitting"),
        ("aaaa", "aaa"),
    ]
    for a, b in pairs:
        j = sa.jaccard(a, b)
        d = sa.dice(a, b)
        assert d == pytest.approx(2 * j / (1 + j), abs=1e-12), (a, b, j, d)


# ---- Argument validation -----------------------------------------

@pytest.mark.parametrize("fn", [sa.jaccard, sa.dice, sa.cosine, sa.overlap])
def test_n_zero_rejected(fn) -> None:
    with pytest.raises(ValueError, match="n must be >= 1"):
        fn("hello", "world", n=0)


# ---- Bytes input -------------------------------------------------

@pytest.mark.parametrize("fn", [sa.jaccard, sa.dice, sa.cosine, sa.overlap])
def test_bytes_input_matches_str_on_ascii(fn) -> None:
    # ASCII bytes widen to Latin-1 codepoints, which equal the str
    # codepoint values for ASCII characters.
    assert fn(b"hello", b"world") == fn("hello", "world")
    assert fn(b"hello", b"hello") == 1.0


# ---- Unicode codepoints ------------------------------------------

def test_non_ascii_codepoints_first_class() -> None:
    # Bigrams of "Müller" and "Mueller" share only "le", "ll", "er".
    # 'Müller'  bigrams: Mü, ül, ll, le, er   → 5 bigrams
    # 'Mueller' bigrams: Mu, ue, el, ll, le, er → 6 bigrams
    # intersect at bigram level: ll, le, er = 3
    # jaccard = 3 / (5 + 6 - 3) = 3 / 8 = 0.375
    assert sa.jaccard("Müller", "Mueller") == pytest.approx(0.375, abs=1e-12)


# ---- Batch -------------------------------------------------------

@pytest.mark.parametrize(
    "fn_pair, sim_fn",
    [
        (sa.jaccard, sa.jaccard_similarities),
        (sa.dice,    sa.dice_similarities),
        (sa.cosine,  sa.cosine_similarities),
        (sa.overlap, sa.overlap_similarities),
    ],
)
def test_batch_matches_per_pair(fn_pair, sim_fn) -> None:
    query = "kitten"
    targets = ["sitting", "kitten", "kit", "biting"]
    batch = sim_fn(query, targets)
    expected = [fn_pair(query, t) for t in targets]
    for got, want in zip(batch.tolist(), expected):
        assert got == pytest.approx(want, abs=1e-12)


@pytest.mark.parametrize(
    "sim_fn",
    [
        sa.jaccard_similarities,
        sa.dice_similarities,
        sa.cosine_similarities,
        sa.overlap_similarities,
    ],
)
def test_batch_returns_float64_ndarray(sim_fn) -> None:
    batch = sim_fn("kitten", ["sitting", "kit"])
    assert isinstance(batch, np.ndarray)
    assert batch.dtype == np.float64
    assert batch.shape == (2,)


@pytest.mark.parametrize(
    "sim_fn",
    [
        sa.jaccard_similarities,
        sa.dice_similarities,
        sa.cosine_similarities,
        sa.overlap_similarities,
    ],
)
def test_batch_rejects_str_or_bytes_targets(sim_fn) -> None:
    with pytest.raises(TypeError, match="not a single str/bytes"):
        sim_fn("query", "single")
    with pytest.raises(TypeError, match="not a single str/bytes"):
        sim_fn("query", b"single")


@pytest.mark.parametrize(
    "sim_fn",
    [
        sa.jaccard_similarities,
        sa.dice_similarities,
        sa.cosine_similarities,
        sa.overlap_similarities,
    ],
)
def test_batch_accepts_generator(sim_fn) -> None:
    def gen():
        yield "sitting"
        yield "kit"
    batch = sim_fn("kitten", gen())
    assert batch.shape == (2,)


def test_batch_n_kwarg_propagates() -> None:
    # n is keyword-only on every entry point.
    batch = sa.jaccard_similarities("hello", ["hello", "help"], n=3)
    assert batch.shape == (2,)
    assert batch[0] == 1.0
    assert batch[1] == pytest.approx(0.25, abs=1e-12)
