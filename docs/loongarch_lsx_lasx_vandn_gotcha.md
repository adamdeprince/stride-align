# LoongArch LSX/LASX `vandn.v` documentation gotcha

**TL;DR.** On LoongArch, `__lsx_vandn_v(a, b)` (and `__lasx_xvandn_v`)
computes `~a & b` — the same semantics as Intel's `_mm_andnot_si128(a,
b)`. This is **not** what the LoongArch ISA reference's instruction
mnemonic "VANDN" suggests when read against the Power/ARM `vandc` /
`bic` convention (which is `a & ~b`). Anyone porting SIMD code from
those ISAs by mnemonic intuition will produce code that compiles, links,
and silently returns wrong answers.

We hit this when porting the bit-parallel Levenshtein + Damerau-
Levenshtein SIMD kernels to `linux_loongarch64_lsx` and `_lasx`. Every
`andnot_` call did the opposite operation, the score-delta mask was
inverted, and small inputs returned negative distances. The fix was a
single-line operand swap once we'd built a probe binary and run it on
real hardware. Three hours of debugging would have been zero if the
manual had simply written out the operation in its first paragraph.

## What our wrapper now does

```cpp
// LSX vandn(x, y) = ~x & y (Intel-style), contrary to the LoongArch
// ISA reference's mnemonic name; same semantics as Intel andnot_si128.
static Vec andnot_(Vec a, Vec b) { return __lsx_vandn_v(a, b); }
```

LASX is the same convention: `__lasx_xvandn_v(a, b) = ~a & b`.

## What the LoongArch documentation says (and why it's confusing)

The LoongArch Vol. 1 ISA reference describes `vandn.v vd, vj, vk` as
"vector AND-NOT". The mnemonic "AND-NOT" is reused across ISAs but
with two distinct meanings:

| ISA | Mnemonic | Operation |
| --- | --- | --- |
| Intel SSE / AVX | `andn` / `andnotps` / `_mm_andnot_si128(a, b)` | `~a & b` |
| ARM NEON | `bic` (bit clear) `vbicq(a, b)` | `a & ~b` |
| PowerPC AltiVec | `vandc(a, b)` / `vec_andc(a, b)` | `a & ~b` |
| LoongArch LSX | `vandn.v vd, vj, vk` | **`~vj & vk` (Intel-style)** |
| LoongArch LASX | `xvandn.v xd, xj, xk` | **`~xj & xk` (Intel-style)** |

The LoongArch ISA reference does state the formula correctly in the
operation pseudocode (`VR[vd] ← ~VR[vj] & VR[vk]`), but the human
reading the mnemonic — especially someone arriving from a Power or ARM
background — will guess wrong. The "AND-NOT" label suggests `vj`
remains AND'd with the *NOT* of `vk`, like Power's `vandc` or ARM's
`bic`.

GCC's intrinsic header `lsxintrin.h` documents
`__lsx_vandn_v(__m128i _1, __m128i _2)` simply as
`(__m128i)__builtin_lsx_vandn_v((v16i8)_1, (v16i8)_2)` — no operand
ordering explanation at all. Same for LASX.

## How to verify

A two-line probe on any LoongArch box:

```cpp
#include <lsxintrin.h>
#include <cstdio>
int main() {
  __m128i a = __lsx_vreplgr2vr_d(0xFFFFFFFFFFFFFFFFLL);
  __m128i b = __lsx_vreplgr2vr_d(0x1LL);
  __m128i r = __lsx_vandn_v(a, b);
  std::printf("vandn(0xFF.., 0x1) lane 0 = 0x%lx\n",
              (unsigned long)__lsx_vpickve2gr_du(r, 0));
  // Intel-style ~a & b -> 0x0
  // Power-style a & ~b -> 0xFFFFFFFFFFFFFFFE
  return 0;
}
```

Build with `g++ -mlsx`, run on the target. We observe `0x0`,
confirming Intel-style semantics.

## Recommendation

If you maintain a portable SIMD wrapper layer:

1. **Don't trust mnemonic naming across ISAs.** Always check the
   operation pseudocode in the ISA reference (or run a probe).
2. **Pick one convention for `andnot_(a, b)` in your wrapper.** This
   project follows Intel's: `andnot_(a, b) = ~a & b`. Power and ARM
   wrappers swap operands at the intrinsic call (`vec_andc(b, a)`,
   `vbicq_u64(b, a)`). LoongArch passes through unchanged.
3. **Add a smoke test.** Any nontrivial SIMD kernel that uses
   `andnot_` will produce wrong answers on day one if the operand
   order is wrong; a single `andnot_(0xFF..F, 1)` unit test catches
   the entire class of bug.

## See also

- `src/cpp/levenshtein_simd_ops.hpp` — the LsxOps / LasxOps bundles,
  where the comment refers back to this doc.
- LoongArch Reference Manual, "Volume 1 — Basic Architecture",
  section "Vector AND-NOT instructions" (`vandn.v` / `xvandn.v`).
- Intel Intrinsics Guide: `_mm_andnot_si128` (Intel-style).
- ARM C Language Extensions: `vbicq_u64` (Power/ARM style).
- AltiVec Programming Interface Manual: `vec_andc` (Power-style).
