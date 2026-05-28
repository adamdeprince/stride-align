# TODO: wide-token path for traceback / path / cigar variants

**Status:** parked.
**Goal:** lift the hard rejection in `prepare_alignment` and let
`smith_waterman_path`, `needleman_wunsch_path`, the corresponding
`_path_info` / `_cigar` entries, and any `_traceback`-style variant
accept the same wide-token inputs (ndarray itemsize ≥ 2 byte; unicode
with >256 distinct codepoints) that the score-only path handles today.

## Why this is hard

The score-only path was tractable for Phase B because:

* The Farrar SW/NW kernel reduces to scoring; only the final cell
  value matters, so the inner DP stays unchanged and we just swap
  in a wider `Cell` type (`int16/int32/int64`).
* The query profile is keyed by the *target* alphabet, so the
  wide-token preprocess just needs to enumerate distinct target
  values — which the Phase B `wide_*` builders already do.

Traceback adds two things the score-only path doesn't have:

1. **A trace table** the size of `query_size * target_size` (one
   entry per DP cell), recording the direction the cell came from
   (diagonal / up / left). The score-only kernels deliberately
   discard this.
2. **A backtrack pass** that walks from the max-score cell (SW) or
   bottom-right (NW) back to the origin, emitting CIGAR ops.

Both pieces are written today against `std::uint8_t` token streams
and a packed-byte trace cell layout. Wide tokens force two changes:

* The trace cell layout has to choose between staying packed (lose
  some token info we may need on tie-breaks) or widening to match
  the cell type (4x memory at int32, 8x at int64). For large wide
  inputs the trace table dominates memory; a 70 000 × 70 000
  int64-cell trace table is 39 GB.
* The backtrack reads `query[i]` / `target[j]` against the wide
  token vector to decide match-vs-mismatch on diagonal moves. That
  read is templated against `std::uint8_t` today (`std::span<const
  std::uint8_t>`) and has to switch to `std::span<const Token>`
  with the same shape we already use in the score-only `wide_*`
  kernels.

## Plan (one possible slice)

1. **Audit** `src/cpp/preprocess.hpp` `prepare_alignment` for ndarray
   rejection sites; lift them in favor of a new
   `prepare_traceback_alignment_wide<Token>` that mirrors the wide
   score preprocess.
2. **Wide trace cell layout.** Pick a packing — proposal: keep the
   trace cell as `uint8_t` (3 bits direction + 5 bits flags), since
   the cell type (`int16/32/64`) drives only the score buffers, not
   the trace.
3. **Templatize the traceback kernels.** `profile_traceback::linear_path`
   et al. currently assume `std::uint8_t` tokens. Hoist the token type
   to a template parameter and instantiate for `uint8/16/32/64`.
4. **Wire through the 9 fixed-kernel backends.** Same pattern as
   Phase B 2.7 / 2.8: per-backend `smith_waterman_path` etc. detect
   ndarray ≥ 2 byte / unicode > 256 distinct and dispatch to the
   wide traceback. Power/VSX is still skipped pending the RAM
   replacement.
5. **Drop or invert `test_path_rejects_ndarray`.** It currently
   pins the reject; once wide traceback lands, replace it with
   parametrized tests that exercise the new path on uint16 / uint32 /
   uint64 ndarrays and large-alphabet unicode.
6. **Memory guard.** Reject inputs whose trace table would exceed a
   user-configurable cap (default 1 GiB?) with a clear "this would
   need N GiB; pass score-only or split your inputs" message rather
   than letting `bad_alloc` happen at depth.

## Scope per slice

A first slice lands one SIMD backend (sse41) with full test coverage,
then propagates the same pattern to the remaining 7 fixed-kernel
backends one at a time. The scalable-kernel backends (SVE/SVE2/RVV)
fall back to the fixed-kernel wide traceback via the same dispatch
trick we use for wide score today (`_FIXED_KERNEL_WIDE_PRIORITY` in
[__init__.py](../src/stride_align/__init__.py)).

## Related

- [project_wide_token_farrar.md](../memory) — parallel score-only wide
  plan that landed as Phase B 2.1 → 2.10.
- [src/cpp/preprocess.hpp](../src/cpp/preprocess.hpp) -- current
  rejection point.
- [tests/test_ndarray.py::test_path_rejects_ndarray](../tests/test_ndarray.py)
  -- the rejection guard to invert when this lands.
