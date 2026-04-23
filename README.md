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
result = stride_align.smith_waterman_path("ACCGT", "CCG")
bench_result = stride_align.smith_waterman_path("ACCGT", "CCG", width=64)
object_result = stride_align.needleman_wunsch_path(
    [frozenset({1}), frozenset({2})],
    [frozenset({1}), frozenset({3})],
)

print(score)
print(result.score, result.aligned_query, result.aligned_target, result.operations)
print(bench_result.score)
print(object_result.aligned_query, object_result.aligned_target)
```

Traceback outputs preserve the paired fast-path type:

- `str` inputs return aligned `str`
- `bytes` inputs return aligned `bytes`
- sequence/object inputs return aligned `tuple` values with `None` gaps

Pass `width=8`, `16`, `32`, or `64` to force a wider kernel than automatic selection.
