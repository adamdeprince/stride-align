# stride-align

**Faster than rapidfuzz. Faster than parasail. One import away.**

**Languages:** [English](README.md) · [简体中文](README.zh-CN.md)

`stride-align` provides SIMD-accelerated fuzzy string matching and sequence
alignment for Python, R, and DuckDB, with first-class Unicode/CJK support. The
Python package also includes phonetic encoding and time-series distance. Its
native packages dispatch to the best compatible backend for x86, ARM,
LoongArch, POWER, and RISC-V, with a portable fallback.

It also provides work-alike imports for four popular libraries:
`import stride_align.rapidfuzz as rapidfuzz` replaces `rapidfuzz`,
`from stride_align.thefuzz import fuzz, process` replaces TheFuzz,
`import stride_align.parasail as parasail` replaces `parasail-python`, and
`import stride_align.jellyfish as jellyfish` replaces `jellyfish` —
existing code moves to stride-align by changing one import.
(New code should prefer the native `stride_align` API.)

The full feature list, every supported algorithm, and per-backend
detail live in the API reference under
[`docs/api/`](docs/api/README.md), with LLM-friendly bundles at
[`llms.txt`](llms.txt) and [`llms-full.txt`](llms-full.txt).

The alignment data plane is also available as a
[host-neutral C++20 library](docs/cpp-core.md). It can be configured without
Python or nanobind and consumed as the CMake target `stride_align::core`:

```sh
cmake -S . -B build/core -G Ninja \
  -DSTRIDE_ALIGN_BUILD_PYTHON=OFF \
  -DSTRIDE_ALIGN_BUILD_CPP_TESTS=ON
cmake --build build/core
ctest --test-dir build/core
```

DuckDB support is delivered as a sideloadable extension with native
`stride_*` pair, vector, ranking, cdist, path, phonetic, DTW, and substitution-
matrix SQL functions plus per-CPU packages. Usage, downloads, and build
instructions are documented in
[`bindings/duckdb/`](bindings/duckdb/README.md).

R support is delivered as the `stridealign` package. It exposes the native
Python API's pair, batch, selection, cdist, alignment, matrix, phonetic, and
DTW capabilities through ordinary R objects, and selects a compatible SIMD
backend when the package loads. Build, install, and API examples are documented in
[`bindings/r/`](bindings/r/README.md).

Instead of giving you a lecture, we're going to learn by doing.
Let's dive right into how it works.

## Getting Started

```bash
pip install stride-align
```

**POWER8 / ppc64le and Loongson / LoongArch64 users:** architecture wheels
are available from the project's
[self-hosted Python downloads](https://distribution.goblinreactor.com/stride-align/python/).
POWER users can follow the [POWER8 installation](#power8-installation);
LoongArch users must also choose between the old-world and new-world binary
worlds described under [LoongArch installation](#loongarch-installation).

### Simple example

First, just a disclaimer: I'm not using religious texts here to push
an agenda - for this demo I need multiple largish public domain
documents that have the same meaning but are phrased differently. The
Bible just happens to fit that demo requirement freakishly well.

Imagine we have two sentences - let's use the first sentence in
Genesis for this:

In the American Standard Version we have: "In the beginning God
created the heavens and the earth."

In the King James Version we have: "In the beginning God created the
heaven and the earth."

We can see with our eyes there's a difference - heavens vs heaven.
But how do we quantify this difference? We'd use this little bit of
code:

```python
import stride_align as sa

print(sa.smith_waterman_normalized_score(
      "In the beginning God created the heavens and the earth.",
      "In the beginning God created the heaven and the earth."))
```

When we run this it prints:

```python
0.9907407407407407
```

Normalized scores are between `0` and `1`. A score of `1` means the
inputs are an exact match under the default scoring model. Scores near
`0` mean the inputs have little in common, though Smith-Waterman may
still find small local matches inside otherwise unrelated strings.

Now let's change the text and see what happens to the score.

```python
import stride_align as sa

print(sa.smith_waterman_normalized_score(
      "In the beginning God created the heavens and the earth.",
      "The quick brown fox jumped over the lazy dog."))
```

and Python prints

```
0.12222222222222222
```

Starting to get the idea? The more similar the strings, the higher the score.

### Larger example: search

Let's build a bigger example, something that gives us a feel for the
library's performance. You'll probably notice that we switch between
Smith-Waterman and Needleman-Wunsch and may be wondering which to use
when. Use Needleman-Wunsch when you want to compare the whole input
against the whole input. Use Smith-Waterman when you want to find the
best matching region inside larger inputs.

Okay, let's move on to the demo code. You need `requests` for this
part of the demo:

```bash
pip install requests
```

```python
import os, time, requests
import stride_align as sa

if not os.path.exists("kjv.txt"):
    response = requests.get("https://openbible.com/textfiles/kjv.txt")
    response.raise_for_status()
    response.encoding = "utf-8-sig"
    open("kjv.txt", "w", encoding="utf-8").write(response.text)

lines = [line.strip().lower() for line in open("kjv.txt")][2:]

while True:
    if not (query := input("Enter a snippet to match.  Press enter to end.\n")):
        break
    t = time.perf_counter()
    scores = sa.needleman_wunsch_normalized_scores(query.lower(), lines)
    best = int(scores.argmax())
    print()
    print("Score:", float(scores[best]))
    print(lines[best])
    print("Search time: %0.2fms" % ((time.perf_counter() - t) * 1000))
    print()
    print()
```

Now how can we use this? Suppose we have a random Bible verse and
want to know what chapter and verse it comes from. `grep` you say?
Oh, heavens, no: we made a mistake. The verse we have is from a
different translation, say the Catholic Public Domain, and what we
have on our computer is the King James Bible. `grep`'s exact string
matching won't work here. How do we find the chapter and verse? We
search for the "closest" or "most similar" string using `stride-align`,
of course.

In our demo the first part concerns itself with downloading and
caching. The good folks at [Open Bible](https://openbible.com) put
this text where it's HTTP-reachable, but we want to be respectful of
their IT budget so we cache what we download. It's just good
citizenship.

In the next part we load all of the lines into a list. We remove
newlines and make everything lower case because we don't want to get
all fiddly about whether we're holding the shift key.

Lastly that `while True:` loop collects a line of text, presumably the
Bible verse from the Catholic version of the Bible we want to look up
the chapter and verse for, and matches it against all of the lines in
the King James Bible using the batch form of Needleman-Wunsch. It
returns an array of scores. We use `argmax()` to find the best-scoring
line and then print the line associated with that index. Let's try it.

I'm going to use Jeremiah 4:28 from the Catholic Bible - it's actually
quite different from the same verse in the King James Bible. Let's see
what happens ...

```
$ python3 demo2.py
Enter a snippet to match.  Press enter to end.
The earth will mourn, and the heavens will lament from above. For I have spoken, I have decided, and I have not regretted. Neither will I be turned away from it.

Score: 0.3598901098901099
jeremiah 4:28	for this shall the earth mourn, and the heavens above be black: because i have spoken [it], i have purposed [it], and will not repent, neither will i turn back from it.
Search time: 206.51ms

```

... and we found it! And pretty quickly too.

#### The same search in DuckDB

The [runnable DuckDB demo](demo/duckdb_vectorized_lookup.py) loads the corpus
as rows, collects those rows into an ordered candidate list, and passes the
whole list to stride-align's optimized one-to-many top-one function:

```sql
WITH corpus AS (
    SELECT
        list(reference ORDER BY reference, verse) AS reference_values,
        list(verse ORDER BY reference, verse) AS verse_values,
        list(lower(verse) ORDER BY reference, verse) AS normalized_verses
    FROM demo_bible
), matched AS (
    SELECT
        reference_values,
        verse_values,
        stride_extract_best(
            lower(?), normalized_verses, 'needleman_wunsch_normalized'
        ) AS best_match
    FROM corpus
)
SELECT
    reference_values[cast(best_match.index AS BIGINT) + 1] AS reference,
    verse_values[cast(best_match.index AS BIGINT) + 1] AS verse,
    best_match.score AS score
FROM matched;
```

DuckDB owns the corpus table and list aggregation. The batch function prepares
the query once, scores the candidate vector, applies its top-one pruning, and
returns the winning zero-based index. Python only submits the query and fetches
its one result; it never loops over verses.

After [downloading the package](https://distribution.goblinreactor.com/stride-align/duckdb/)
for your platform and CPU, run it like this:

```bash
export STRIDE_ALIGN_DUCKDB_EXTENSION=/absolute/path/to/stride_align.0.6.0.duckdb_extension
python3 demo/duckdb_vectorized_lookup.py bible
```


### Larger example: spell checker

This is a toy spell checker, not a production one. It ignores punctuation,
capitalization, word frequency, proper nouns, and context. The point is to
show the same one-query-against-many-candidates pattern on a familiar task.

```python
import os, sys
import stride_align as sa

paths = ['/usr/share/dict/words',
         '/usr/dict/words',
         '/var/lib/dict/words',
         '/etc/dictionaries-common/words']

for path in paths:
    if os.path.exists(path):
        break
else:
    print("Sorry, I can't find your dictionary", file=sys.stderr)
    exit(1)


words = [line.strip().lower() for line in open(path)]


for line in sys.stdin:
    new_line = []
    for word in line.split():
        scores = sa.needleman_wunsch_normalized_scores(word.lower(), words)
        word = words[int(scores.argmax())]
        new_line.append(word)
    print(' '.join(new_line), flush=True)
```

The first thing this script does is try to find our operating system's
list of correctly spelled words. Its location can vary from
distribution to distribution. Once we've found it, we load it, strip
off newlines and start the act of spell checking.

The spell checking looks a lot like the matching we did before. For
each candidate word, we match it against all of the words in our list
of correctly spelled words, use `argmax()` to find the highest-scoring
candidate, and replace the word with that candidate. We could speed
things up with some optimizations, like not searching for a match for
correctly spelled words, but this is a demo and that optimization is
left as an exercise for the reader.

Let's see how it works!

```bash
$ cat - | python3 demo3.py
this is a demonstrtion of a spel checker
it doesn't matter that I can't spell corectly

this is a demonstration of a spell checker
it doesn't matter that i can't spell correctly
```

#### The same spell checker in DuckDB

The DuckDB mode tokenizes a whole input line, collects the dictionary into a
candidate list, and applies stride-align's batch top-one operation once per
token. The matches return as one ordered line in a single SQL statement. There
is no Python loop around either the tokens or the dictionary:

```sql
WITH dictionary AS (
    SELECT list(word ORDER BY word) AS words
    FROM demo_dictionary
), input_tokens AS (
    SELECT position, token
    FROM unnest(string_split(lower(?), ' '))
         WITH ORDINALITY AS tokens(token, position)
),
matched AS (
    SELECT
        position,
        stride_extract_best(
            token, words, 'needleman_wunsch_normalized'
        ) AS best_match
    FROM input_tokens
    CROSS JOIN dictionary
)
SELECT string_agg(best_match.target, ' ' ORDER BY position)
FROM matched;
```

Use a system dictionary or pass `--dictionary` explicitly:

```bash
python3 demo/duckdb_vectorized_lookup.py spell
```


## Capabilities

The native `stride_align` API covers, in one library, the surface
that's usually split across `parasail`, `rapidfuzz`,
`python-Levenshtein`, `jellyfish`, `editdistance`, and `dtw-python`.

**Edit distance and similarity scorers.** Levenshtein,
Damerau-Levenshtein (both OSA and unrestricted variants), Indel,
Hamming, Jaro, Jaro-Winkler, longest common subsequence / substring,
Ratcliff-Obershelp, Monge-Elkan, character-n-gram Jaccard / Sørensen-
Dice / cosine / overlap. Each algorithm exposes a consistent
variant family: `_score`, `_normalized_score`, `_scores` (one query
× N targets, batched), `_normalized_scores`, `_best`,
`_normalized_best`, `_top_k`. Detail in
[`docs/api/edit-distance.md`](docs/api/edit-distance.md) and
[`docs/api/similarity.md`](docs/api/similarity.md).

**Sequence alignment.** Smith-Waterman (local) and Needleman-Wunsch
(global) DP with linear *or* affine gaps and substitution matrices.
Score-only, normalised, batch-over-targets, top-k, traceback to
alignment path, SAM/BAM-style CIGAR (`_cigar`, `_trace_cigar`,
`_trade_cigar`). A Farrar score-only fast path uses an interleaved
DP layout for the throughput-oriented case. Detail in
[`docs/api/alignment.md`](docs/api/alignment.md).

**All-pairs cdist family.** `cdist`, `cdist_above_threshold`,
`cdist_top_k`, and `cdist_top_k_per_query` over any built-in scorer
(via the `Scorer` enum) or any Python callable. Multi-threaded SIMD
under a released GIL, with closed-form length-difference pruning and
per-pair cutoff push-down into the kernel inner loop. Detail in
[`docs/api/cdist.md`](docs/api/cdist.md).

**Substitution matrices.** Built-in `blosum45` / `50` / `62` / `80`
/ `90` and `pam30` / `70` / `250`, an NCBI-text loader, and a
generic `SubstitutionMatrix` for custom alphabets (case-sensitive
text included). Detail in
[`docs/api/matrices.md`](docs/api/matrices.md).

**Phonetic encoders.** Soundex, Metaphone (Apache Commons or
jellyfish rule families), Double Metaphone (Apache Commons or
Python-package bug-compat), NYSIIS, Match Rating Approach,
Caverphone 2, Cologne Phonetic (Unicode-aware German),
Daitch-Mokotoff Soundex, and Beider-Morse Phonetic Matching. Detail
in [`docs/api/phonetic.md`](docs/api/phonetic.md).

**Dynamic time warping.** `dtw_distances` for one query against many
targets, `int16` / `float32` / `float64`, optional Sakoe-Chiba band,
choice of local metric. Detail in
[`docs/api/dtw.md`](docs/api/dtw.md).

**Compatibility shims.** `stride_align.rapidfuzz`,
`stride_align.thefuzz`, `stride_align.parasail`, and
`stride_align.jellyfish` are work-alike
import replacements for widely used libraries in the space. Detail in
[`docs/api/rapidfuzz-shim.md`](docs/api/rapidfuzz-shim.md) and
[`docs/api/thefuzz-shim.md`](docs/api/thefuzz-shim.md),
[`docs/api/parasail-shim.md`](docs/api/parasail-shim.md), and
[`docs/api/jellyfish-shim.md`](docs/api/jellyfish-shim.md).

The native boundary accepts:

- `bytes` against `bytes`
- `str` against `str` (UCS-1 / UCS-2 / UCS-4, zero-copy — Chinese,
  Japanese, Korean, Arabic, emoji all hit the SIMD path without a
  UTF-8 round-trip)
- sequences of immutable hashable Python objects
- mixed sequence/object inputs where a `str` or `bytes` side is
  treated as a sequence
- NumPy `ndarray` of integer dtype (8 / 16 / 32 / 64 bit)

Direct `bytes` versus `str` pairs raise `TypeError`.

Score-only functions return numeric scores. The normalised variants
return scores between `0` and `1`. Path functions return alignment
result objects with the score, aligned sequences, operations, and
CIGAR-style summaries where available.

## Documentation

| File | Audience | Contents |
| --- | --- | --- |
| [`README.md`](README.md) | new users | this file — installation, quick start, capability overview |
| [`docs/api/`](docs/api/README.md) | application developers | per-surface API reference (edit-distance, similarity, alignment, cdist, matrices, DTW, phonetic, shims) |
| [`llms.txt`](llms.txt) | LLMs / agents | brief index for the [llmstxt.org](https://llmstxt.org/) convention |
| [`llms-full.txt`](llms-full.txt) | LLMs / agents | single-page concatenation of `README.md` + every page under `docs/api/` |
| [`BENCHMARK.md`](BENCHMARK.md) | perf-curious | cross-architecture performance numbers vs `parasail`, `rapidfuzz`, `python-Levenshtein`, `editdistance` |
| [`CHANGELOG.md`](CHANGELOG.md) | upgraders | version history with breaking-change notes |
| [`docs/adding-a-new-algorithm.md`](docs/adding-a-new-algorithm.md) | contributors | the internal kernel + binding pattern |
| [`docs/loongson-build.md`](docs/loongson-build.md) | LoongArch packagers | dual-toolchain (old-world / new-world) build recipe |

Both READMEs and every markdown file in the repo are rendered to
HTML by [`tools/md_to_html.py`](https://github.com/adamdeprince/stride-align/blob/main/tools/md_to_html.py). The long-form
English README is published as `README.html`. The tracked English product
homepage is authored directly at [`html/index.html`](https://github.com/adamdeprince/stride-align/blob/main/html/index.html); the
Simplified Chinese homepage and shared assets are authored under
[`website/`](https://github.com/adamdeprince/stride-align/tree/main/website).
The published site lives in [`html/`](https://github.com/adamdeprince/stride-align/tree/main/html) and is mirrored at
[stride-align.com](https://stride-align.com).

### Cloudflare Workers hosting

The site is configured as an assets-only Cloudflare Worker. Wrangler uploads
the built `html/` tree directly; there is no JavaScript Worker entry point or
runtime binding.

For **Cloudflare Workers Builds**, set this build variable in
**Settings → Build → Build Variables and Secrets**. This disables Cloudflare's
automatic Python dependency detection, which would otherwise see
`pyproject.toml` and try to compile the stride-align native extension:

```text
SKIP_DEPENDENCY_INSTALL=1
```

Use these exact build settings:

```text
Build command:  bash tools/cloudflare_build.sh
Deploy command: npx wrangler deploy --assets ./html
Root directory: /
```

The Cloudflare build script runs `npm ci`, installs only the pure-Python
dependency in [`requirements-site.txt`](requirements-site.txt), and runs the
Markdown renderer. It never installs the stride-align package or invokes a C/C++
compiler.

For local development, use Node.js 22 or newer and install the Python Markdown
build dependency once:

```bash
python3 -m pip install "markdown>=3.5"
npm ci
```

Then build and preview locally, validate the upload without deploying, or deploy
to the configured `stride-align` Worker. Authenticate once before the first
interactive deployment:

```bash
npx wrangler login
npm run site:dev
npm run site:check
npm run site:deploy
```

`site:build` runs [`tools/md_to_html.py`](tools/md_to_html.py). That generator
preserves the tracked, hand-authored `html/index.html` and refreshes the
localized homepage, documentation, styles, benchmark artifacts, and shared
assets around it. Wrangler account selection and any `stride-align.com` custom
domain route remain Cloudflare account configuration, so a dry run cannot
change live DNS or routing.

## API quick-start

The full reference lives under
[`docs/api/`](docs/api/README.md), grouped by surface (edit-distance,
similarity, alignment, all-pairs cdist, substitution matrices, DTW,
phonetic encoders, and the four compatibility shims). This section is
a tour of the most common patterns to get you started.

```python
import stride_align

score = stride_align.needleman_wunsch_score("ACGT", "ACCT")
scores = stride_align.Scores("ACGT", variant="needleman_wunsch").compare(["ACCT", "AGGT"])
result = stride_align.smith_waterman_path("ACCGT", "CCG")
wide_result = stride_align.smith_waterman_path("ACCGT", "CCG", width=64)
object_result = stride_align.needleman_wunsch_path(
    [frozenset({1}), frozenset({2})],
    [frozenset({1}), frozenset({3})],
)

print(score)
print(scores)
print(result.score, result.aligned_query, result.aligned_target, result.operations)
print(wide_result.score)
print(object_result.aligned_query, object_result.aligned_target)
```

Use `Scores(...).compare([...])` or the `*_scores()` functions for one-query
against many-target score workloads. These are the preferred performance APIs
for repeated English/Chinese text comparisons.

Traceback outputs preserve the paired fast-path type:

- `str` inputs return aligned `str`
- `bytes` inputs return aligned `bytes`
- sequence/object inputs return aligned `tuple` values with `None` gaps

Pass `width=8`, `16`, `32`, or `64` to force the internal token/scoring width
instead of using automatic selection.

Some functions expose CIGAR strings, short for "Concise Idiosyncratic
Gapped Alignment Report". CIGAR is the compact alignment-operation notation
used by SAM/BAM tooling. If you want the full formal version, see the
[SAM specification](https://samtools.github.io/hts-specs/SAMv1.pdf).

### Substitution matrices (BLOSUM, PAM)

For protein alignment, `stride_align.matrices` ships the canonical
BLOSUM and PAM substitution matrices. Pass any of them via the
`matrix=` kwarg on `smith_waterman_score`, `needleman_wunsch_score`,
or their `_scores` batch counterparts:

```python
import stride_align
from stride_align.matrices import blosum62, pam250

# Local alignment, NCBI standard BLOSUM62 with affine gaps (open=-11,
# extend=-1). matrix= is mutually exclusive with match_score / mismatch_score.
stride_align.smith_waterman_score(
    "HEAGAWGHEE", "PAWHEAE",
    matrix=blosum62,
    gap_open_score=-11, gap_extend_score=-1,
)

# Batch (1 query × N targets) with profile reuse — the recommended
# path for "score one query against a library".
stride_align.smith_waterman_scores(
    "HEAGAWGHEE",
    ["PAWHEAE", "HEAGAWGHEE", "MEEPS"],
    matrix=pam250, gap_open_score=-14, gap_extend_score=-2,
)

# Custom matrices: parse any NCBI-format text file
custom = stride_align.matrices.SubstitutionMatrix.from_ncbi_text(
    open("/path/to/BLOSUM62").read(),
    name="BLOSUM62",
    gap_open=-11, gap_extend=-1,
)
```

Each built-in `SubstitutionMatrix` exposes its alphabet, matrix data
(`int8` ndarray), and recommended gap defaults via `.gap_score`
(linear), `.gap_open`, and `.gap_extend`. Both linear gaps (`gap_score=`)
and affine gaps (`gap_open_score=` + `gap_extend_score=`) are
supported on the AVX-512 backend; other SIMD backends currently fall
back to the scalar generic kernel for matrix-mode.

The shipped matrix values come from the NCBI BLAST distribution
[`ftp.ncbi.nih.gov/blast/matrices/`](https://ftp.ncbi.nih.gov/blast/matrices/),
which carries the canonical reference scores. The original
publications are:

- **BLOSUM45 / 50 / 62 / 80 / 90** — Henikoff S., Henikoff J.G. (1992).
  *Amino acid substitution matrices from protein blocks*. PNAS
  89(22):10915–10919.
  [doi:10.1073/pnas.89.22.10915](https://doi.org/10.1073/pnas.89.22.10915)
  &nbsp;·&nbsp;
  [PDF (open access)](https://www.pnas.org/doi/pdf/10.1073/pnas.89.22.10915)
- **PAM30 / 70 / 250** — Dayhoff M.O., Schwartz R.M., Orcutt B.C.
  (1978). *A model of evolutionary change in proteins*. In *Atlas of
  Protein Sequence and Structure*, vol. 5, supplement 3, pages 345–352.
  National Biomedical Research Foundation, Washington, D.C. (Book
  chapter; not available online as an open PDF. A widely cited
  follow-on derivation appears in Schwartz R.M., Dayhoff M.O. (1978),
  *Matrices for detecting distant relationships*, same volume,
  pages 353–358.)

### Keyboard typo matrices (real typing-error data)

stride-align also ships example **keyboard confusion matrices** built from
real human typing errors — the Aalto "136 Million Keystrokes" dataset, used
with the authors' permission. Each scores how likely it is that one
character was typed when another was meant, so a plausible slip (`teh` for
`the`) aligns as a near-match instead of a flat mismatch:

```python
import stride_align as sa
from stride_align.matrices import keyboard

sa.smith_waterman_score("teh", "the", matrix=keyboard.qwerty)
```

**Orientation — the query is the misspelled side.** Like every stride-align
matrix these are `m[a][b]` with `a` = query and `b` = target, and here
**`a` is the character actually typed (the mistake) and `b` is the character
that was intended (the correction)**. So pass the misspelled / user-entered
string as the query and the canonical / dictionary string as the target.

Unlike symmetric BLOSUM/PAM, these matrices are **asymmetric**, so
orientation matters. If your pipeline has it the other way round — the query
is the *correct* string — flip the matrix with NumPy's `.T` (transpose),
which swaps the axes to `m[intended][misspelled]`:

```python
import numpy as np
from stride_align.matrices import SubstitutionMatrix, keyboard

fwd = keyboard.qwerty
rev = SubstitutionMatrix(
    name=fwd.name + ".T", alphabet=fwd.alphabet,
    matrix=np.ascontiguousarray(fwd.matrix.T),   # .T swaps query/target
    gap_score=fwd.gap_score, wildcard=fwd.wildcard,
)
```

Build your own from a `{(typed, intended): count}` mapping with
`keyboard.from_confusion_counts(...)`. The shipped matrices are derived
log-odds artifacts, not the raw keystroke data; see `NOTICE` and
`docs/keyboard-matrix-external-sources.md` for attribution and the scope of
the permission.

### rapidfuzz compatibility (drop-in shim)

Replace one import line and most rapidfuzz code keeps working:

```python
# Before:
# import rapidfuzz

# After:
import stride_align.rapidfuzz as rapidfuzz

# fuzz: full token-ratio family, scores in [0, 100]
rapidfuzz.fuzz.ratio("hello", "hallo")                  # 80.0
rapidfuzz.fuzz.WRatio("foo bar baz", "foo bar")         # 90.0
rapidfuzz.fuzz.token_set_ratio("the cat", "cat the")    # 100.0

# distance: classes with distance / normalized / similarity methods,
# plus editops / opcodes for Levenshtein.
rapidfuzz.distance.Levenshtein.distance("kitten", "sitting")          # 3
rapidfuzz.distance.JaroWinkler.normalized_similarity("MARTHA", "MARHTA")
rapidfuzz.distance.Levenshtein.editops("kitten", "sitting")
# -> Editops([Editop(tag='replace', src_pos=0, dest_pos=0), ...], src_len=6, dest_len=7)

# process: extract / extractOne / cdist
rapidfuzz.process.extract("hello", ["hallo", "world", "helo"], limit=2)
# -> [('helo', 88.88, 2), ('hallo', 80.0, 0)]

# utils: default_process (matches upstream bit-exactly, does NOT
# collapse internal whitespace runs)
rapidfuzz.utils.default_process("Hello, World!")        # 'hello  world'
```

Known divergences: the `partial_ratio` family inherits stride-align's
Phase D.3 conservative-underestimate — never overshoots upstream, but
can underestimate by a few points on pairs where rapidfuzz finds a
shifted optimal window. `Levenshtein.distance` does not yet support
the `weights=(insert, delete, replace)` kwarg.

### TheFuzz compatibility (work-alike facade)

TheFuzz keeps an older integer-scored API and two-element extraction
tuples. The dedicated facade preserves those conventions while using
stride-align's native kernels:

```python
# Before:
# from thefuzz import fuzz, process

# After:
from stride_align.thefuzz import fuzz, process

fuzz.ratio("this is a test", "this is a test!")       # 97
fuzz.token_sort_ratio("fuzzy wuzzy was a bear", "wuzzy fuzzy was a bear")
# -> 100
process.extractOne("cowboys", ["New York Jets", "Dallas Cowboys"])
# -> ('Dallas Cowboys', 90)
```

All ten TheFuzz 0.22.1 scorers, `extract*`, and `dedupe` are covered.
The direct scorers also accept arbitrary sequences of hashable Python
objects through stride-align's shared compact-token encoder. See
[`docs/api/thefuzz-shim.md`](docs/api/thefuzz-shim.md) for exact
preprocessing, rounding, result-shape, and custom-scorer behavior.
The `partial_ratio` family inherits the same rare shifted-window
conservative underestimate documented for the RapidFuzz shim above.

### parasail compatibility (drop-in shim)

Replace one import line and most parasail code keeps working:

```python
# Before:
# import parasail

# After:
import stride_align.parasail as parasail

# Same parasail signature: (s1, s2, open, extend, matrix)
# Gap penalties are positive numbers (BLAST convention:
# cost(N) = open + (N-1)*extend).
r = parasail.sw_trace("HEAGAWGHEE", "PAWHEAE", 11, 1, parasail.blosum62)
print(r.score)               # int
print(r.cigar.decode)        # bytes, e.g. b'2=1X3='
print(r.traceback.query)     # 'HEAGAWGHEE' aligned with gaps
print(r.traceback.ref)       # 'PAWHEAE'    aligned with gaps
print(r.traceback.comp)      # '|.| ||'-style match annotation

# matrix_create + stats
m = parasail.matrix_create("ACGT", 2, -1)
r = parasail.sw_stats("ACGTAC", "ACATAC", 5, 2, m)
print(r.matches, r.similar, r.length)

# The 2000+ kernel-suffix variants (sw_striped_avx2_16, nw_scan_64,
# sw_trace_diag_sat, ...) all alias to the matching core entry —
# stride-align picks the kernel based on score range and hardware.
parasail.sw_striped_avx2_16("ACGT", "ACGT", 5, 2, m)
```

Known divergences: SW with multiple optimal alignments may pick a
different path than upstream parasail (both score-correct); the
`sg_qb`/`sg_qe`/`sg_qb_de` style semi-global mode selectors and the
`dnafull` / `nuc44` matrices are not yet provided.

### Edit-distance scorers

Beyond Smith-Waterman and Needleman-Wunsch, `stride-align` exposes
six unit-cost edit-distance and similarity metrics — each with its
own SIMD-batched code path:

```python
import stride_align

# Levenshtein (Myers 1999 bit-parallel) — inserts, deletes, substitutes
stride_align.levenshtein_score("kitten", "sitting")               # -> 3
stride_align.levenshtein_normalized_score("kitten", "sitting")    # -> 0.571...
stride_align.levenshtein_scores("kitten", ["kit", "sitting"])     # -> ndarray[int64]

# Optional `score_cutoff` (rapidfuzz convention): bail early per-target,
# results that exceed the cutoff come back as `cutoff + 1`.
stride_align.levenshtein_scores(query, targets, score_cutoff=3)

# Damerau-Levenshtein (OSA-restricted, Hyyrö 2002) — adds adjacent
# transposition at unit cost. This is what rapidfuzz exposes as
# OSA.distance and is what most callers asking for
# "Damerau-Levenshtein" actually want.
stride_align.damerau_levenshtein_score("ab", "ba")                # -> 1

# True Damerau-Levenshtein — the unrestricted form, where one
# character may participate in more than one edit. Slower (no
# bit-parallel kernel yet) but matches rapidfuzz.distance.DamerauLevenshtein
# exactly. Diverges from OSA on overlapping transpositions, e.g.
# "ca" -> "abc": OSA=3, true-DL=2.
stride_align.true_damerau_levenshtein_score("ca", "abc")          # -> 2

# Indel — Levenshtein restricted to insertions and deletions, no
# substitutions. Equivalent to |a| + |b| - 2 * LCS(a, b). Bit-
# parallel Allison-Dix (1986) inner loop.
stride_align.indel_score("kitten", "sitting")                     # -> 5

# Hamming — count of positions where two equal-length strings differ.
# Cutoff variant bails the byte loop once mismatches exceed the cap.
stride_align.hamming_score("100", "110")                          # -> 1

# Jaro / Jaro-Winkler — similarities in [0, 1]; Winkler adds a
# capped prefix bonus.
stride_align.jaro_similarity("martha", "marhta")                  # -> 0.944...
stride_align.jaro_winkler_similarity("martha", "marhta")          # -> 0.961...
```

The batch variants (`*_scores`, `*_similarities`) pack one target
per SIMD lane on every supported backend:

- x86: SSE4.1 / AVX2 / AVX-512 / AVX10-256 / AVX10-512
- ARM: NEON (Linux + macOS), SVE / SVE2
- LoongArch: LSX / LASX
- PowerPC: VSX

For Lev / OSA, patterns up to 64 chars run a single-word Myers;
65–256 chars use the multi-word kernel (W=2/3/4). Indel and OSA
fall back to scalar bit-parallel for patterns >64 (multi-word
generalization deferred); true-DL is scalar DP only.

### Longest Common Subsequence + Substring

Two related but distinct dynamic programs, both shipped:

```python
import stride_align as sa

# Longest Common Subsequence — characters need not be contiguous.
# "ABCBDAB" and "BDCAB" share "BCAB" (length 4).
sa.lcs_length("ABCBDAB", "BDCAB")                    # -> 4

# Closed-form relation to Indel distance: indel = |a| + |b| - 2·LCS.
sa.indel_score("kitten", "sitting") == \
    len("kitten") + len("sitting") - 2 * sa.lcs_length("kitten", "sitting")
# -> True

# Longest Common Substring — characters MUST be contiguous.
sa.lcs_substring_length("ABCBDAB", "BDCAB")          # -> 2
sa.lcs_substring("ABCBDAB", "BDCAB")                 # -> "AB"

# Result type matches inputs: bytes in, bytes out.
sa.lcs_substring(b"hello world", b"world hello")     # -> b"hello"

# Codepoint engine — non-ASCII is first-class.
sa.lcs_substring("Müller", "Mueller")                # -> "ller"
```

Both DPs are scalar `O(m·n)` time with two rolling rows for
`O(min(m,n))` (subsequence) or `O(|b|)` (substring) space. When
multiple substrings tie at the maximum length, the first occurrence
in `a` is returned (matches `str.find` convention).

### Ratcliff-Obershelp similarity

The algorithm Python's `difflib.SequenceMatcher().ratio()` ships,
which `rapidfuzz` does not — recursive longest-matching-substring
split, summed match lengths divided by total length:

```python
import stride_align as sa

sa.ratcliff_obershelp_similarity("kitten", "sitting")
# -> 0.6153846153846154

# Bit-exact with difflib at autojunk=False (we have no junk
# character heuristic):
import difflib
sa.ratcliff_obershelp_similarity("ABCBDAB", "BDCAB") == \
    difflib.SequenceMatcher(None, "ABCBDAB", "BDCAB", autojunk=False).ratio()
# -> True

# Batch form: one query against many targets, returned as
# ndarray[float64].
sa.ratcliff_obershelp_similarities("kitten",
                                    ["sitting", "kitten", "kit"])
# -> array([0.61538462, 1.        , 0.66666667])
```

Not commutative — the inner longest-common-substring tiebreak
(`earliest in a, then earliest in b`) means the recursion splits
leftover ranges differently for `(a, b)` vs `(b, a)`, and the total
match length can differ. Faithful to difflib, which has the same
property; `sa.ratcliff_obershelp_similarity("ABCBDAB", "BDCAB")`
gives `0.333…` while the reverse gives `0.667…`. Pin both
directions if your tests need an order-independent metric.

### N-gram set similarity

Four metrics over character n-gram **multisets** (each n-gram counted
with multiplicity), keyword-only `n=` (default 2 — character bigrams):

```python
import stride_align as sa

# Jaccard: |A ∩ B| / |A ∪ B|
sa.jaccard("ABCBDAB", "BDCAB")                  # -> 0.25

# Sørensen-Dice: 2 * |A ∩ B| / (|A| + |B|)
sa.dice("ABCBDAB", "BDCAB")                     # -> 0.4

# Overlap coefficient: |A ∩ B| / min(|A|, |B|)
sa.overlap("ABCBDAB", "BDCAB")                  # -> 0.5

# Cosine over multiset frequency vectors: ⟨A, B⟩ / (‖A‖ · ‖B‖)
sa.cosine("ABCBDAB", "BDCAB")                   # -> ~0.5303

# Trigrams.
sa.jaccard("hello", "help", n=3)                # -> 0.25

# Batch — query multiset built once and reused across targets.
sa.jaccard_similarities("kitten", ["sitting", "kitten", "kit"])
# -> array([0.25, 1.0, 0.111...])
```

All four metrics are symmetric and bounded in `[0, 1]`. Identity
convention: both inputs empty (or both shorter than `n`) → `1.0`;
one empty → `0.0`. Dice and Jaccard satisfy the closed-form
relation `D = 2·J / (1 + J)`.

### Token-ratio family (rapidfuzz `fuzz.*` parity)

Drop-in replacements for the `rapidfuzz.fuzz.*` token-ratio API,
returning values in `[0, 1]` (multiply by 100 for rapidfuzz's
`[0, 100]` convention). The base ratio is `sa.indel_normalized_score`
— algebraically identical to `rapidfuzz.fuzz.ratio / 100` (both
reduce to `2 · LCS / (|a| + |b|)`).

```python
import stride_align as sa

# Token sort: split on whitespace, sort, join, compute the ratio.
sa.token_sort_ratio("fuzzy wuzzy bear", "bear wuzzy fuzzy")     # -> 1.0

# Token set: set intersection + per-side differences, max of three
# pairwise ratios.
sa.token_set_ratio("the quick brown fox", "the quick brown dog") # -> ~0.895

# Partial ratio: best match of the shorter string within the longer
# (sliding-window + LCS-substring candidate).
sa.partial_ratio("apple", "an apple a day")                     # -> 1.0
sa.partial_ratio("java language",
                 "python programming language")                 # -> ~0.818

# Token-sort / token-set combined with partial ratio.
sa.partial_token_sort_ratio("apple bear", "an apple and a bear") # -> 1.0
sa.partial_token_set_ratio("the cat",     "a cat sat down")      # -> 1.0

# rapidfuzz's weighted blend.
sa.WRatio("fuzzy wuzzy was a bear", "wuzzy fuzzy was a bear")    # -> 1.0

# Case-insensitive: pass a processor callable.
sa.token_sort_ratio("FOO BAR", "bar foo", processor=str.lower)   # -> 1.0
```

`token_set_ratio` and `partial_token_set_ratio` follow rapidfuzz's
convention of returning `0.0` when either side has no tokens after
whitespace splitting. The implementations are pure Python on top of
stride-align's own kernels — no third-party code is imported into
the production path.

### Monge-Elkan multi-token similarity

Classic record-linkage hybrid (Monge & Elkan, 1996). For each token
in `s1`, find the best-matching token in `s2` under a configurable
inner similarity, then average across `s1`'s tokens. Asymmetric by
definition — pass `symmetric=True` to average both directions when
an order-independent score is wanted.

```python
import stride_align as sa

# Default inner is Jaro.
sa.monge_elkan("paul johnson", "paul jones")      # -> ~0.94

# Asymmetric: |s1| tokens drive the average.
sa.monge_elkan("paul",         "paul johnson")    # -> 1.0
sa.monge_elkan("paul johnson", "paul")            # -> 0.5

# Symmetric variant.
sa.monge_elkan("paul",         "paul johnson",
               symmetric=True)                    # -> 0.75

# Inner similarity selection.
sa.monge_elkan("hello world", "hallo world",
               inner="jaro_winkler")              # boost common prefixes
sa.monge_elkan("hello world", "hallo world",
               inner="levenshtein_ratio")         # bit-parallel Levenshtein
sa.monge_elkan("a b c", "a c d",
               inner=lambda x, y: 1.0 if x == y else 0.0)  # custom callable

# Preprocessor (e.g. case-insensitive).
sa.monge_elkan("PAUL JOHNSON", "paul Johnson",
               processor=str.lower)               # -> 1.0
```

Returns `1.0` when both inputs have no tokens after whitespace
splitting (vacuously identical); `0.0` when exactly one side has no
tokens. The implementation is pure Python on top of stride-align's
Jaro / Jaro-Winkler / Levenshtein / Indel kernels — no new C++
kernels and no third-party code in the production path.

### Phonetic encoders

For name matching, deduplication, and search-as-you-type, `stride-align`
ships the full standard phonetic-encoder family. Each encoder maps a
string to a short code such that names that *sound* similar share a
code, regardless of spelling:

```python
import stride_align as sa

# American Soundex (Russell & Odell, 1918). 4-character code.
sa.soundex("Robert")                                     # -> "R163"
sa.soundex("Rupert")                                     # -> "R163"
sa.soundex_equal("Robert", "Rupert")                     # -> True

# Metaphone (Lawrence Philips, 1990) — two-letter and longer
# spec-correct variants. The published 1990 spec and the popular
# jellyfish library disagree on a handful of edge cases; the variant
# kwarg picks the rule family.
sa.metaphone("Schmidt")                                  # -> "SKMTT"  (PHILIPS, spec)
sa.metaphone("Schmidt", variant=sa.MetaphoneVariant.JELLYFISH)  # -> "SXMTT"
sa.metaphone_equal("Schmidt", "Smith")                   # -> False

# Double Metaphone (Lawrence Philips, 2000) — primary and alternate
# codes; the alternate captures plausible non-English pronunciations.
# COMMONS is the faithful Apache Commons Codec port; PYTHON is bug-
# compat with the metaphone PyPI package.
sa.double_metaphone("Schwartz")                          # -> ("XRTS", "XFRTS")
sa.double_metaphone("Hugh")                              # -> ("H", "")
sa.double_metaphone("Hugh",
    variant=sa.DoubleMetaphoneVariant.PYTHON)            # -> ("HH", "")

# NYSIIS (Taft, 1970). More discriminative than Soundex for English
# names — "Watkins" / "Wilkins" / "Wilkinson" don't collide.
sa.nysiis("Watkins"), sa.nysiis("Wilkins")               # -> ("WATCAN", "WALCAN")

# Match Rating Approach (Moore, Western Airlines, 1977). A codex plus
# a pairwise comparator with length-difference + rating-threshold rules.
sa.match_rating_codex("Christopher")                     # -> "CHRPHR"
sa.match_rating_compare("Robert", "Rupert")              # -> True

# Caverphone 2.0 (Hood, 2004). Fixed-length 10-character code,
# right-padded with '1'. Designed for late-19th-century New Zealand
# electoral rolls but widely applied to general English-language
# name matching.
sa.caverphone("Stevenson")                               # -> "STFNSN1111"

# Cologne Phonetic / Kölner Phonetik (Postel, 1969). German-language
# encoder that maps letters to digits 0-8 with context-sensitive rules
# for C, X, D, T, P. Umlauts and ß preprocess to their Latin-letter
# equivalents so callers don't have to NFKD-fold first.
sa.cologne_phonetic("Müller")                            # -> "657"
sa.cologne_phonetic("Schmidt")                           # -> "862"

# Daitch-Mokotoff Soundex (Daitch & Mokotoff, 1985). Six-digit
# Soundex tuned for Slavic and Yiddish surnames. The leading letter
# is encoded (not preserved verbatim); multi-character clusters like
# 'sch', 'tsch', 'rz' fire before any single-letter rule; several
# rules emit '|'-joined alternative codes via branching.
sa.daitch_mokotoff("LEWINSKY")                           # -> "876450"
sa.daitch_mokotoff("Goldman")                            # -> "583660"
sa.daitch_mokotoff("AUERBACH")                           # -> "097400|097500"
sa.daitch_mokotoff("AUERBACH", branching=False)          # -> "097400"

# Beider-Morse Phonetic Matching (Beider & Morse, 2008). Multi-
# language phonetic encoder returning a '|'-separated set of plausible
# pronunciation codes across European languages, optimised for family
# names. stride-align ships the GENERIC name-type only — the broad
# general-purpose rule set; the Ashkenazi and Sephardic rule sets from
# the upstream Apache Commons Codec distribution are not included.
sa.beider_morse("Renault")
# -> "rinD|rinDlt|rina|rinalt|rino|rinolt|rinu|rinult"
sa.beider_morse("Renault", rule_type=sa.BmpmRuleType.EXACT)
# -> "renau|renault|reno|renolt"
sa.beider_morse("Müller", rule_type=sa.BmpmRuleType.EXACT)
# -> "mQler|muler"
sa.beider_morse("d'ortley", rule_type=sa.BmpmRuleType.EXACT)
# -> "(ortlaj|ortlej)-(dortlaj|dortlej)"   (d' prefix handler)
```

The first seven encoders are dispatched through the same byte-
extraction helper, accept `str` and `bytes` inputs interchangeably,
and skip non-letter / non-ASCII codepoints before encoding — pre-
normalise with `unicodedata.normalize("NFKD", s)` if you want accent
folding. Cologne Phonetic re-encodes `str` inputs through UTF-8 so its
ß / Ä / Ö / Ü preprocessing fires correctly. Beider-Morse ships its
GENERIC rule data (the 63 `gen_*.txt` files from Apache Commons Codec)
as package resources loaded once at first call via
`importlib.resources`, runs the language guesser plus a rule-based
phonetic engine entirely in C++, and returns a `|`-separated UTF-8
string of phonetic codes. Cross-checked against the canonical Apache
Commons Codec reference data and the `jellyfish`, `metaphone`, and
`doublemetaphone` PyPI packages.

### Dynamic Time Warping

For aligning numeric sequences whose timing or speed varies — audio
signals, gesture / sensor traces, financial time series —
`stride-align` exposes Dynamic Time Warping with optional Sakoe-Chiba
band:

```python
import numpy as np
import stride_align as sa

q = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
t = np.array([1.0, 2.0, 2.5, 4.0, 5.0])

# Default distance follows the dtype:
#   float32 / float64 -> L2-squared, (x - y)^2
#   int16             -> L1, |x - y|  (audio convention)
sa.dtw(q, t)                                              # -> 0.25

# Sakoe-Chiba band: int radius or fraction of max(|q|, |t|).
sa.dtw(q, t, window=2)
sa.dtw(q, t, window=0.2)

# Explicit distance.
sa.dtw(q.astype(np.int16), t.astype(np.int16), distance="l1")

# Batch.
sa.dtw_distances(q, [t, t * 2, t + 0.5], window=2)
```

Inputs must be NumPy `ndarray` with matching dtype (`float32`,
`float64`, or `int16` — the natural audio dtype). Other dtypes and
non-ndarray inputs raise `TypeError`.

### `cdist`, `cdist_above_threshold`, `cdist_top_k`, `cdist_top_k_per_query`

For all-pairs scoring across two lists of strings, `stride-align`
ships three matrix-style entry points:

```python
qs = ["kitten", "sitting", "kit"]
ts = ["kitten", "kit", "sitting", "biting"]

# Full N×M similarity matrix — ndarray[float64] (similarity scorers)
# or ndarray[int64] (distance scorers).
sa.cdist(qs, ts, scorer=sa.Scorer.JARO)

# Streaming filter — yields only pairs whose similarity exceeds the
# threshold. Workers feed a bounded queue; the caller drains it.
# Length pruning + per-pair cutoff push-down into the kernel skip
# most of the work at high thresholds.
for score, q, t in sa.cdist_above_threshold(
    qs, ts, scorer=sa.Scorer.LEVENSHTEIN_NORMALIZED, threshold=0.7,
):
    ...

# Top-k by score — returns at most k highest-scoring (or lowest, for
# distance scorers) (score, query, target) tuples. Heaps are
# per-thread; a shared atomic global-min bound lets the per-pair
# cutoff push-down lift the prune threshold as work progresses.
sa.cdist_top_k(qs, ts, scorer=sa.Scorer.JARO, k=10)

# Top-k targets PER QUERY, yielded as a generator. Differs from
# cdist_top_k (which returns the k highest pairs globally) by keeping
# a separate top-k heap per query. With pruning=True, the worst-in-
# heap score adapts as scoring progresses and targets whose closed-
# form length-difference upper bound on similarity can't beat it
# are skipped before the kernel runs — a big win on workloads with
# wide length variation.
for query, top in sa.cdist_top_k_per_query(
    qs, ts, scorer=sa.Scorer.LEVENSHTEIN_NORMALIZED, k=5, pruning=True,
):
    # top is [(score, target), ...] sorted descending
    ...

# Smith-Waterman and Needleman-Wunsch on cdist. The SW / NW scorers
# accept the same scoring parameters as the per-pair calls; both
# raw-score and normalised-similarity variants are available. The
# dispatch happens via a Python-level ThreadPoolExecutor over rows
# because the SW / NW per-row kernels release the GIL.
sa.cdist(qs, ts, scorer=sa.Scorer.SMITH_WATERMAN,
         match_score=2, mismatch_score=-1, gap_score=-1)   # int64
sa.cdist(qs, ts, scorer=sa.Scorer.SMITH_WATERMAN_NORMALIZED)  # float64 in [0, 1]
sa.cdist(qs, ts, scorer=sa.Scorer.NEEDLEMAN_WUNSCH,
         gap_open_score=-5, gap_extend_score=-1)            # int64, can be negative
sa.cdist(qs, ts, scorer=sa.Scorer.NEEDLEMAN_WUNSCH_NORMALIZED)
```

At high thresholds the pruning is dramatic — see the cross-arch
table in [BENCHMARK.md](https://stride-align.com/BENCHMARK.html) (the `cdist pruning` rows).
Loongson LASX in particular flips the expected ranking against
Tiger Lake AVX-512 at T=0.99; the comparison report lives at
[docs/loongson-vs-tiger-lake-cdist-2026-05-24.md](docs/loongson-vs-tiger-lake-cdist-2026-05-24.md).

See [BENCHMARK.md](https://stride-align.com/BENCHMARK.html) for full cross-architecture numbers.

## Optimizations and Benchmarks

Careful attention has been, and continues to be, paid to `stride-align`'s
performance story. The library includes SIMD optimization for a variety of
common targets, including x86, Arm, and LoongArch.

**Intel AVX-512 versus Parasail (2026-07-18).** The current pinned sweep ran
on a Granite Rapids Xeon 6975P-C with GCC 16.1 and Parasail 1.3.4. It covers
80 directly comparable combinations: English and Chinese inputs, linear and
affine gaps, 16- and 32-bit score widths, `1:1` and `1:many` shapes, and seven
score/path/CIGAR variants. Ratios are Parasail median runtime divided by
stride-align median runtime, so values above 1 mean stride-align is faster.

| AVX-512 group | Rows won | Geomean | Median |
| --- | ---: | ---: | ---: |
| Overall | **79 / 80** | **1.682x** | 1.656x |
| Score-only | **47 / 48** | **1.559x** | 1.558x |
| Path / CIGAR | **32 / 32** | **1.883x** | 1.737x |

The strongest variant geomeans are Needleman-Wunsch path at 2.356x,
Needleman-Wunsch CIGAR at 2.341x, and Smith-Waterman score at 1.989x. The
single losing AVX-512 row is Chinese, linear-gap, `1:1` Needleman-Wunsch
score at width 16 (0.874x). The complete methodology, AVX2 comparison,
per-variant tables, and five worst rows are in [BENCHMARK.md](BENCHMARK.md).
The canonical raw data is [benchmark.csv](benchmark.csv), with an immutable
[dated snapshot](benchmarks/intel-avx512-parasail-2026-07-18.csv).

**rapidfuzz shim full-surface bench (v0.5.0).** Across 108 workloads
covering every public entry point of `stride_align.rapidfuzz` (10
`fuzz` methods, 8 distance classes × 4 methods, `process.cdist` +
`process.extract` with several scorers), the cross-architecture
geomeans against upstream `rapidfuzz` 3.14.5 are:

| Host | Backend | Geomean | Wins / Ties / Losses |
| --- | --- | ---: | ---: |
| Mac M4 Max | `macos_arm64_neon` | 1.34x | 95 / 5 / 8 |
| Intel AWS | `x86_avx10_512` | 1.02x | 68 / 13 / 27 |
| Loongson | `linux_loongarch64_lasx` | 49.17x | 108 / 0 / 0 |

(Ratio = upstream / shim, > 1.0 means shim is faster.) The Mac M4 Max
backend wins or ties 100 of 108 workloads (geomean 1.34x); its 8
losses are the bit-exact `partial_token_ratio` recipe (kept exact
rather than fast), a few tiny-string Hamming cases, and the
multithreaded `process.cdist` / `process.extract` throughput
harnesses. Intel lands just past parity (1.02x) with a wider tail in
those same cdist/throughput and token-composite workloads. Loongson
is a clean sweep because upstream rapidfuzz ships no LoongArch wheel.
Mac and Intel re-measured 2026-06-17; Loongson last measured
2026-06-10 (pre the Jaro and token-ratio work, so unchanged or
better today).

**Striped Smith-Waterman / Needleman-Wunsch kernel update (2026-07).** The
score-only local-SW and NW kernels were reworked. Measured throughput of the new
kernels against the previous ones (native microbench, 1 query × 8 targets,
length 1024, match +8 / mismatch −9), 2026-07-16:

- **Affine-gap local SW** — a faster prefix lazy-F correction: **roughly
  +30–45%** across AVX2, AVX-512, and NEON. (The win narrows at the exact length
  where the specialized exact-fill path engages; one AVX2 i32 corner there is a
  small regression — every other config wins.)
- **Needleman-Wunsch** — unchanged (±1%).
- **Linear-gap local SW** — a **correctness fix**. The previous fast path used an
  unsound early-exit that could silently under-report scores on structured inputs
  (post-mortem and reproducible counter-examples in
  [`docs/known-issue-bounded-lazy-f-scan.md`](docs/known-issue-bounded-lazy-f-scan.md)).
  The replacement is exact everywhere; a `deferred` correction recovers most of
  the lost speed where a sound early-exit is impossible. The residual cost is
  architecture-dependent:

  | Backend | Host | Correctness cost at length 1024 |
  | --- | --- | ---: |
  | AVX2 | AMD (naamah) | ~3% (13.8 → 13.4 Gcells/s) |
  | NEON | Apple M4 | ~2% (4.35 → 4.26 Gcells/s) |
  | AVX-512 | Intel (avx10) | ~22% (16.0 → 12.4 Gcells/s) |
  | LASX | Loongson | ~0% (5.33 → 5.34 Gcells/s, Python-timed batch) |

  On the narrow-SIMD backends the exact answer is nearly free; on AVX-512 no
  sound early-exit exists, so `deferred` and the naive full scan both land at
  ~12.5 Gcells/s — correctness costs ~20% there, a trade taken deliberately for a
  library whose scores are meant to be exact.

**LoongArch / Loongson.** The Loongson optimization story is especially
telling: for the checked benchmark case -- English text, 16-bit score width,
score-only Smith-Waterman -- the LASX backend is 16x faster than the generic
backend and **22.4x** faster than Parasail.

If you are a researcher using Loongson servers and benefiting from this
speedup, citations, bug reports, benchmark cases, and tiny inexpensive Chinese
souvenirs are appreciated. Tea, calligraphy bookmarks, paper-cut ornaments,
Chinese knot charms, panda keychains, and small dragon desk objects are all
welcome. Please do not send anything expensive or anything that requires
customs paperwork.

See [complete benchmarks](https://stride-align.com/BENCHMARK.html).

## Native Microbench

For perf profiling without Python frames or benchmark orchestration, configure a
native x86 microbench build:

```bash
nanobind_dir="$(.venv/bin/python -m nanobind --cmake_dir)"
cmake -S . -B build/perf \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DSTRIDE_ALIGN_BUILD_MICROBENCH=ON \
  -DSTRIDE_ALIGN_PERF_SYMBOLS=ON \
  -DPython_EXECUTABLE=.venv/bin/python \
  -Dnanobind_DIR="$nanobind_dir"
cmake --build build/perf --target stride_align_x86_microbench
build/perf/stride_align_x86_microbench --backend avx2 --shape 1:many --pass english --width 16
python tools/x86_microbench_regression.py \
  --binary build/perf/stride_align_x86_microbench \
  --cpu 2 \
  --backends avx2,avx512bwvl \
  --shapes 1:1,1:many \
  --passes english,chinese \
  --widths 16,32 \
  --write-json /tmp/stride-align-x86-microbench.json
.venv/bin/python tools/pinned_benchmark_sweep.py \
  --output-dir /tmp/stride-align-pinned \
  --cpu 2 \
  --iterations 15 \
  --warmups 3
```

`STRIDE_ALIGN_PERF_SYMBOLS=ON` keeps nanobind modules unstripped and adds debug
symbols plus frame pointers while preserving `-O3`.

The checked-in native microbench baseline lives at
`benchmarks/x86_microbench_baseline.json`. Treat it as a local guardrail with a
loose threshold, not as a cross-machine SLA.


## POWER8 installation

Runtime-tested Linux `ppc64le` wheels for POWER8 and newer processors are
self-hosted for CPython 3.12, 3.13, and 3.14. Choose the wheel matching the
running interpreter:

```bash
PY=$(python3 -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')
python3 -m pip install \
  https://distribution.goblinreactor.com/stride-align/python/v0.6.0/stride_align-0.6.0-${PY}-${PY}-linux_ppc64le.whl
```

The complete artifact list and SHA-256 checksums are on the
[self-hosted Python download page](https://distribution.goblinreactor.com/stride-align/python/).
The wheel contains the generic and VSX implementations and selects the best
compatible backend when imported.


## LoongArch installation

LoongArch wheels are **not on PyPI** (PyPI doesn't index the
`linux_loongarch64` platform tag). The project serves them directly from its
Vermont distribution host:

| Resource | URL |
| --- | --- |
| Download page | `https://distribution.goblinreactor.com/stride-align/python/` |
| Versioned wheel prefix | `https://distribution.goblinreactor.com/stride-align/python/v0.6.0/` |

The versioned directory is immutable. The download page lists every wheel,
its validation status, checksums, and a machine-readable manifest.

Install NumPy from your distro first (loongarch64 NumPy wheels are
sparse on PyPI, and the distro one is usually ABI-compatible with
the rest of the system):

```bash
sudo apt install python3-numpy
PY=$(python3 -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')
```

### Old-world vs new-world: what to pick

LoongArch hardware runs in one of two mutually incompatible binary
**worlds**. They differ in two things:

1. **Which dynamic loader the executable references** (this is the
   filename baked into the ELF header at link time).
2. **Which glibc ABI version the binary depends on**.

A wheel built for one world will not load on the other — the loader
filename doesn't exist on the wrong side, and the symbols would
mismatch even if it did. We ship one wheel per world:

| World | Loader | glibc | Typical hosts | Wheel build tag |
| --- | --- | --- | --- | --- |
| **Old-world** | `/lib64/ld.so.1` | 2.28-era | Stock Kylin, original Loongson distros | `1.oldworld` |
| **New-world** | `/lib64/ld-linux-loongarch-lp64d.so.1` | ≥ 2.36 | Recent LoongArch distros whose Python uses the new loader | `1.newworld` |

Both wheels are statically linked against libstdc++ / libgcc so the
only thing separating them is the loader / glibc ABI.

**Pick the right one from the loader embedded in your Python executable**:

```bash
PYTHON_LOADER=$(readelf -l "$(readlink -f "$(command -v python3)")" |
  sed -n 's/.*Requesting program interpreter: \(.*\)]/\1/p')
case "$PYTHON_LOADER" in
  */ld-linux-loongarch-lp64d.so.1) WORLD=newworld ;;
  */ld.so.1) WORLD=oldworld ;;
  *) echo "Unrecognized Python loader: $PYTHON_LOADER" >&2; exit 1 ;;
esac
echo "$WORLD"
```

Use the wheel matching the result. Do not select a world from the processor
model or the mere presence of a loader file: Python's embedded loader and
glibc ABI are what matter.

### Old-world wheel

```bash
pip install \
  https://distribution.goblinreactor.com/stride-align/python/v0.6.0/stride_align-0.6.0-1.oldworld-${PY}-${PY}-linux_loongarch64.whl
```

### New-world wheel

Use this package only when the detection command above prints
`new-world`; a new-world distribution supplies the matching loader and glibc
as one system runtime.

```bash
pip install \
  https://distribution.goblinreactor.com/stride-align/python/v0.6.0/stride_align-0.6.0-1.newworld-${PY}-${PY}-linux_loongarch64.whl
```

### Other notes

Prebuilt LoongArch64 wheels are available for Python 3.12, 3.13,
and 3.14 in both worlds. The build details
(toolchains, loader validation, static C++ runtime) live in
[docs/loongson-build.md](docs/loongson-build.md). If you're on a
different Python or want to build from source, `pip install
stride-align` falls back to the PyPI source distribution and
compiles the LSX/LASX kernels locally.


## Citations

If you use my software in your research, please cite me.

```bibtex
@software{deprince_stride_align,
  author       = {DePrince, Adam},
  title        = {stride-align: Fast Smith-Waterman and Needleman-Wunsch alignment for Python},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/adamdeprince/stride-align},
  note         = {Python/C++ library for sequence and string alignment}
}
```

## References

stride-align's SIMD Smith-Waterman / Needleman-Wunsch kernels take algorithmic
ideas from **parasail** (Jeff Daily's SIMD pairwise-alignment library — the
striped-profile layout and lazy-F gap correction). If you use stride-align in
published research, please also cite parasail:

```bibtex
@article{daily2016parasail,
  author  = {Daily, Jeff},
  title   = {Parasail: SIMD C library for global, semi-global, and local pairwise sequence alignments},
  journal = {BMC Bioinformatics},
  year    = {2016},
  volume  = {17},
  number  = {1},
  pages   = {1--11},
  doi     = {10.1186/s12859-016-0930-z}
}
```
