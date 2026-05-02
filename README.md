# stride-align

`stride-align` is a Python alignment library implemented in C++23 with nanobind.

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

## API

```python
import stride_align

score = stride_align.needleman_wunsch_score("ACGT", "ACCT")
scores = stride_align.Scores("ACGT", variant="needleman_wunsch").compare(["ACCT", "AGGT"])
result = stride_align.smith_waterman_path("ACCGT", "CCG")
bench_result = stride_align.smith_waterman_path("ACCGT", "CCG", width=64)
object_result = stride_align.needleman_wunsch_path(
    [frozenset({1}), frozenset({2})],
    [frozenset({1}), frozenset({3})],
)

print(score)
print(scores)
print(result.score, result.aligned_query, result.aligned_target, result.operations)
print(bench_result.score)
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

Pass `width=8`, `16`, `32`, or `64` to force a wider kernel than automatic selection.

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
