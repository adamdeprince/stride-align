# TheFuzz facade external-source audit

This audit covers `src/stride_align/thefuzz/`, its focused tests, and the
generic-sequence extension to the native partial-ratio dispatcher. Native
stride-align code remains Apache-2.0; the two compatibility references below
are MIT-licensed and Apache-2.0 compatible. Their license texts are retained
in `NOTICE`, which is included in wheel metadata.

## TheFuzz 0.22.1

- **Source:** https://github.com/seatgeek/thefuzz, release 0.22.1
- **License:** MIT, Copyright (c) 2014 SeatGeek
- **Used for:** the public module/function/signature surface; integer rounding;
  `force_ascii` and `full_process` composition; Unicode aliases; process
  defaults, tuple shapes, and scorer lowering behavior; `dedupe` semantics.
- **Adaptation:** the facade is an independent stride-align-backed adapter,
  but its compatibility structure and observable legacy conventions are
  adapted from TheFuzz under the MIT license.
- **Runtime dependency:** none. TheFuzz is imported only by the optional
  differential section of `tests/test_thefuzz_shim.py`.

## RapidFuzz 3.14.5

- **Source:** https://github.com/rapidfuzz/RapidFuzz, release 3.14.5
- **License:** MIT, Copyright © 2020-present Max Bachmann; Copyright © 2011
  Adam Cohen
- **Used for:** TheFuzz delegates scoring and extraction to RapidFuzz, so its
  public behavior is part of the compatibility target. The facade retains the
  generic-sequence normalization convention (single-character strings become
  code points; other objects use 64-bit hash tokens), whitespace-token
  behavior, token-set score operation order, and ranking by unrounded scores.
  RapidFuzz is also an optional black-box differential oracle.
- **Adaptation:** compatibility adapters are expressed in original Python and
  dispatch scoring to stride-align's kernels; no RapidFuzz binary or source is
  shipped. The MIT text is nevertheless retained because source was consulted
  for these edge-case conventions.
- **Runtime dependency:** none.

## Explicit GPL exclusion

FuzzyWuzzy (`github.com/seatgeek/fuzzywuzzy`) is GPL-2.0. Its source, tests,
fixtures, and expected-output vectors were not read, copied, translated, or
used. The facade targets its MIT-licensed successor TheFuzz only. No claim in
the API documentation changes that exclusion.

## Files affected

- `src/stride_align/thefuzz/{__init__,fuzz,process,utils}.py`
- `src/cpp/partial_ratio_dispatch.hpp`
- `include/stride_align/partial_ratio.hpp`
- `tests/test_thefuzz_shim.py`
- `docs/api/thefuzz-shim.md`
- `NOTICE`, `README.md`, `README.zh-CN.md`, `CHANGELOG.md`, `pyproject.toml`
