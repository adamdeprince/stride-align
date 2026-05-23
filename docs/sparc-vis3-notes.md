# SPARC T5 / VIS3 Backend Notes

Target: Oracle SPARC T5 (sun4v, 3600 MHz) running Solaris 11.4.

## VIS feature inventory

`isainfo -kv` reports: `vis3c vis3b ... vis3 ... vis2 vis`.

| Instruction | VIS level | Available on T5? | Use here |
| --- | --- | --- | --- |
| `fpadd16` / `fpsub16` | VIS1 | yes | per-lane add/sub for i16 cells |
| `fpadd32` / `fpsub32` | VIS1 | yes | per-lane for i32 cells |
| `fpadds16` / `fpsubs16` (saturating) | VIS3 | yes | `saturating_add`, the `add_sentinel` shortcut |
| `fcmpgt16` / `fcmpeq16` (returns int) | VIS1 | yes | predicate input for cmask/bshuffle |
| `cmask16` (4-bit -> GSR.MASK) | VIS3 | yes | mask vector reconstruction |
| `bshuffle` | VIS2 | yes | per-byte select |
| `faligndata` (cross-register byte shift) | VIS1 | yes | shift_left_zero |
| `fpmax16` / `fpmin16` (partitioned max/min) | **VIS4** | **NO** | emulated, see below |
| `fpaddus16` (unsigned saturating add) | VIS4 | NO | not usable (Wozniak bias trick) |

## The partitioned-max gap

T5 only has VIS3, not VIS4, so `fpmax16` is not in hardware. Running the
`__builtin_vis_fpmax16` builtin produces an Illegal Instruction trap.

GCC's vector codegen synthesises max via:

```
fcmpgt16  %a, %b, %mask     ; 1 cycle, scalar 4-bit mask
fzero     %z                 ; constant, hoist out of loop
fone      %f                 ; constant, hoist out of loop
cmask16   %mask              ; VIS3, writes GSR.MASK
bshuffle  %f, %z, %vmask     ; expand 4-bit to v4hi mask
fxor      %a, %b, %ab        ; a XOR b
fand      %vmask, %ab, %r    ; r = mask & (a XOR b)
fxor      %r, %b, %r         ; r = (mask & (a XOR b)) XOR b == (mask ? a : b)
```

That's six in-loop ops per partitioned max. NEON does it with one
`vmaxq_s16`; AVX2 with one `vpmaxsw`. T5 paying 6x per max is the dominant
limit on this backend's SIMD speedup.

## Sentinel-aware add via fpadds16

Mirror of the SVE2 `svqadd_s16` optimisation. For
`add_sentinel(lhs, rhs, INT16_MIN)` where `rhs <= 0` (gap_open /
gap_extend), `fpadds16` collapses the 3-op `cmpeq + add + sel` guard to a
single instruction: `INT_MIN + (<= 0)` saturates to `INT_MIN` (sentinel
preserved); a normal value plus a small negative stays in range.

Wired as `SimdOps::saturating_add` and picked up by the scalable kernel's
`add_sentinel_negative_rhs` helper.

## Lazy-F prefix carry

i16 vector is 4 lanes -> 2 log-steps (shift by 1, then 2). i32 vector is 2
lanes -> 1 log-step. `__builtin_shuffle` with compile-time indices lowers to
`faligndata` via the GSR.ALIGN scratch.

## Standalone benchmark, no integration

The standalone Farrar SW kernel (`~/sparc_explore/standalone_sw.cc`) shows:

| Query / target | Scalar | VIS3 | Speedup |
| --- | ---: | ---: | ---: |
| 128 / 128   | 45.9 Mc/s | 57.3 Mc/s | 1.25x |
| 256 / 256   | 47.2 Mc/s | 61.2 Mc/s | 1.30x |
| 512 / 512   | 48.3 Mc/s | 63.9 Mc/s | 1.32x |
| 1024 / 1024 | 47.7 Mc/s | 65.2 Mc/s | 1.37x |
| 2048 / 2048 | 47.7 Mc/s | 65.9 Mc/s | 1.38x |

The affine path (`~/sparc_explore/standalone_affine.cc`) shows a different
story: GCC's autovectoriser on the cell-by-cell Gotoh loop competes
well with the hand-coded Farrar-striped VIS3 kernel because every Farrar
max paid the 6-instruction emulation. For affine the autovectorised scalar
runs about 100 Mc/s on a 1024x1024 input; hand-coded Farrar with naive
lazy-F runs about 50 Mc/s. The integrated build's prefix-carry + exact-fill
kernels are expected to recover most of that gap; numbers TBD after the
GCC 14 / Python toolchain finishes building.

## Practical perf ceiling

T5 is a 2013 CMT design with 2 vCPUs in this VM and a slow single-thread
IPC. Modern NEON gets ~2.5 Gc/s on the same algorithm; AVX2 ~3 Gc/s. T5's
SIMD ceiling is roughly 65 Mc/s for linear SW, ~50-100 Mc/s for affine SW
depending on which kernel wins. This is an architecture-support port, not
a competitive-performance port.
