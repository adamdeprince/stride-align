# stride-align

`stride-align` is a [blazing fast library](BENCHMARK.md) to tell you how "similar" two strings are.
It does this by implementing the Smith-Waterman and Needleman-Wunsch
algorithms. Instead of giving you a lecture, we're going to learn by
doing. Let's dive right into how it works.

## Installation

```bash
pip install stride-align
```

On Loongson systems, install NumPy from your Linux distribution before
installing `stride-align`, and grab the LoongArch64 wheel from the
GitHub release instead of PyPI (PyPI does not yet accept the
`linux_loongarch64` or `manylinux_2_38_loongarch64` platform tags):

```bash
sudo apt install python3-numpy

PY=$(python3 -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')
pip install \
  https://github.com/adamdeprince/stride-align/releases/download/v0.1.0/stride_align-0.1.0-${PY}-${PY}-linux_loongarch64.whl
```

Prebuilt LoongArch64 wheels are available for Python 3.10, 3.11, 3.12,
3.13, and 3.14. If you are on a different Python (or just want to
build from source), `pip install stride-align` falls back to the
source distribution on PyPI, which compiles the LSX/LASX kernels
locally.

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
Oh, heavens, no, we made a mistake. The verse we have is from a
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


Now let's do another demo: spell checking.

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

The first thing this script does is try to find our operating
system's list of correctly spelled words. Its location can vary from
distribution to distribution. Once we've found it, we load it, strip off
newlines and start the act of spell checking.

The spell checking looks a lot like the matching we did before. For each
candidate word, we match it against all of the words in our list of
correctly spelled words, use `argmax()` to find the highest-scoring
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


## Details

The current scaffold provides:

- Needleman-Wunsch score-only alignment
- Needleman-Wunsch alignment with traceback
- Smith-Waterman score-only alignment
- Smith-Waterman alignment with traceback
- A backend layout that matches the specialization pattern used in `massive-speedup`
- CPU/backend detection and Python-side backend dispatch

The native boundary accepts:

- `bytes` against `bytes`
- `str` against `str`
- sequences of immutable hashable Python objects
- mixed sequence/object inputs where a `str` or `bytes` side is treated as a sequence

Direct `bytes` versus `str` pairs raise `TypeError`.

The current implementations are generic dynamic-programming kernels with preprocessing
that serializes Python inputs into 8, 16, 32, or 64-bit token streams. SIMD-specialized
backends can replace the backend translation units later without changing the Python API.

Score-only functions return numeric scores. The normalized variants return
scores between `0` and `1`. Path functions return alignment result objects
containing the score, aligned values, operations, and CIGAR-style summaries
where available.

## API

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
against many-target score workloads. That path prepares the query/profile once
and is the preferred performance API for repeated English/Chinese text
comparisons.

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

## Optimizations and Benchmarks

Careful attention has been, and continues to be, paid to `stride-align`'s
performance story. The library includes SIMD optimization for a variety of
common targets, including x86, Arm, and LoongArch.

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

See [complete benchmarks](BENCHMARKS.md).

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
