#pragma once

// SIMD primitive bundles consumed by levenshtein_simd::myers_batch_single_word.
// Each bundle is gated by the matching __target__ pragma the backend was
// compiled with; backends that don't enable the corresponding ISA simply
// won't reference its bundle.

#include <cstdint>

#if defined(__SSE4_1__) || defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#if defined(__loongarch_sx)
#include <lsxintrin.h>
#endif

#if defined(__loongarch_asx)
#include <lasxintrin.h>
#endif

#if defined(__VSX__) || defined(__POWER8_VECTOR__)
#include <altivec.h>
// Altivec headers leak macro names like `vector`, `bool`, `pixel`.
// `vector` is the worst offender — undefine it so it doesn't clash
// with std::vector. We use `__vector` explicitly throughout.
#ifdef vector
#undef vector
#endif
#ifdef bool
#undef bool
#endif
#ifdef pixel
#undef pixel
#endif
#endif

namespace stride_align::levenshtein_simd {

#if defined(__SSE4_1__)

struct SseOps {
  static constexpr std::size_t lanes = 2;
  using Vec = __m128i;

  static Vec set1(std::uint64_t v) {
    return _mm_set1_epi64x(static_cast<long long>(v));
  }
  static Vec zero() { return _mm_setzero_si128(); }
  static Vec and_(Vec a, Vec b) { return _mm_and_si128(a, b); }
  static Vec or_(Vec a, Vec b) { return _mm_or_si128(a, b); }
  static Vec xor_(Vec a, Vec b) { return _mm_xor_si128(a, b); }
  static Vec not_(Vec a) {
    return _mm_xor_si128(a, _mm_set1_epi64x(static_cast<long long>(-1LL)));
  }
  static Vec add(Vec a, Vec b) { return _mm_add_epi64(a, b); }
  static Vec sub(Vec a, Vec b) { return _mm_sub_epi64(a, b); }
  // shift left by 1 = add to self
  static Vec shl1(Vec a) { return _mm_add_epi64(a, a); }
  // Per-lane: all-ones if a == b, else zero (64-bit lane).
  static Vec cmpeq(Vec a, Vec b) { return _mm_cmpeq_epi64(a, b); }
  // (~a) & b
  static Vec andnot_(Vec a, Vec b) { return _mm_andnot_si128(a, b); }
  static Vec gather64(const std::uint64_t* base, const std::uint64_t* indices) {
    return _mm_set_epi64x(
        static_cast<long long>(base[indices[1]]),
        static_cast<long long>(base[indices[0]]));
  }
  static Vec load_aligned(const std::uint64_t* data) {
    return _mm_load_si128(reinterpret_cast<const __m128i*>(data));
  }
  static void store_aligned(std::uint64_t* dst, Vec v) {
    _mm_store_si128(reinterpret_cast<__m128i*>(dst), v);
  }
  // Per-lane unsigned 64-bit `a > b`. SSE4.1 lacks cmpgt_epi64; emulate via
  // (a - b) sign bit + zero check (works for any uint64 values).
  static Vec gt_u64(Vec a, Vec b) {
    const Vec diff = _mm_sub_epi64(a, b);
    const Vec sign_lsb = _mm_srli_epi64(diff, 63);
    const Vec is_ge = _mm_cmpeq_epi64(sign_lsb, _mm_setzero_si128());
    const Vec is_eq = _mm_cmpeq_epi64(diff, _mm_setzero_si128());
    return _mm_andnot_si128(is_eq, is_ge);
  }
  // Logical right-shift by 63: each 64-bit lane becomes 0 or 1 (the top bit).
  static Vec shr63(Vec a) { return _mm_srli_epi64(a, 63); }
  // True iff every bit of `a` is zero. Used by the cutoff early-exit
  // path to test whether any lane is still live.
  static bool is_zero(Vec a) { return _mm_testz_si128(a, a) != 0; }
  // Per-lane left/right shift by a per-lane count. SSE4.1 has no
  // variable 64-bit shift (VPSLLVQ is AVX2+), so emulate via
  // extract/insert. Two lanes means two scalar shifts — cheap enough
  // for the Jaro window-mask computation that needs it.
  static Vec shl_var_u64(Vec a, Vec shifts) {
    const auto a0 = static_cast<std::uint64_t>(_mm_extract_epi64(a, 0));
    const auto a1 = static_cast<std::uint64_t>(_mm_extract_epi64(a, 1));
    const auto s0 = static_cast<std::uint64_t>(_mm_extract_epi64(shifts, 0));
    const auto s1 = static_cast<std::uint64_t>(_mm_extract_epi64(shifts, 1));
    return _mm_set_epi64x(
        static_cast<long long>(a1 << s1),
        static_cast<long long>(a0 << s0));
  }
  static Vec shr_var_u64(Vec a, Vec shifts) {
    const auto a0 = static_cast<std::uint64_t>(_mm_extract_epi64(a, 0));
    const auto a1 = static_cast<std::uint64_t>(_mm_extract_epi64(a, 1));
    const auto s0 = static_cast<std::uint64_t>(_mm_extract_epi64(shifts, 0));
    const auto s1 = static_cast<std::uint64_t>(_mm_extract_epi64(shifts, 1));
    return _mm_set_epi64x(
        static_cast<long long>(a1 >> s1),
        static_cast<long long>(a0 >> s0));
  }
};

#endif  // __SSE4_1__

#if defined(__AVX2__)

struct Avx2Ops {
  static constexpr std::size_t lanes = 4;
  using Vec = __m256i;

  static Vec set1(std::uint64_t v) {
    return _mm256_set1_epi64x(static_cast<long long>(v));
  }
  static Vec zero() { return _mm256_setzero_si256(); }
  static Vec and_(Vec a, Vec b) { return _mm256_and_si256(a, b); }
  static Vec or_(Vec a, Vec b) { return _mm256_or_si256(a, b); }
  static Vec xor_(Vec a, Vec b) { return _mm256_xor_si256(a, b); }
  static Vec not_(Vec a) {
    return _mm256_xor_si256(a, _mm256_set1_epi64x(static_cast<long long>(-1LL)));
  }
  static Vec add(Vec a, Vec b) { return _mm256_add_epi64(a, b); }
  static Vec sub(Vec a, Vec b) { return _mm256_sub_epi64(a, b); }
  static Vec shl1(Vec a) { return _mm256_add_epi64(a, a); }
  static Vec cmpeq(Vec a, Vec b) { return _mm256_cmpeq_epi64(a, b); }
  static Vec andnot_(Vec a, Vec b) { return _mm256_andnot_si256(a, b); }
  static Vec gather64(const std::uint64_t* base, const std::uint64_t* indices) {
    const __m256i idx = _mm256_load_si256(reinterpret_cast<const __m256i*>(indices));
    return _mm256_i64gather_epi64(
        reinterpret_cast<const long long*>(base), idx, 8);
  }
  static Vec load_aligned(const std::uint64_t* data) {
    return _mm256_load_si256(reinterpret_cast<const __m256i*>(data));
  }
  static void store_aligned(std::uint64_t* dst, Vec v) {
    _mm256_store_si256(reinterpret_cast<__m256i*>(dst), v);
  }
  // Per-lane unsigned 64-bit `a > b`. cmpgt_epi64 is signed; flip the
  // sign bit of both inputs to recover the unsigned ordering. Used in
  // Myers' wide-add carry detection where operands can span the full
  // uint64 range.
  static Vec gt_u64(Vec a, Vec b) {
    const Vec sign = _mm256_set1_epi64x(
        static_cast<long long>(0x8000000000000000ULL));
    return _mm256_cmpgt_epi64(
        _mm256_xor_si256(a, sign),
        _mm256_xor_si256(b, sign));
  }
  static Vec shr63(Vec a) { return _mm256_srli_epi64(a, 63); }
  static bool is_zero(Vec a) { return _mm256_testz_si256(a, a) != 0; }
  // AVX2 has native variable 64-bit shifts (VPSLLVQ / VPSRLVQ).
  static Vec shl_var_u64(Vec a, Vec shifts) {
    return _mm256_sllv_epi64(a, shifts);
  }
  static Vec shr_var_u64(Vec a, Vec shifts) {
    return _mm256_srlv_epi64(a, shifts);
  }
};

#endif  // __AVX2__

#if defined(__AVX512F__)

// 16-lane uint32 sibling of Avx512Ops, used by the bit-parallel batch
// kernels when the query fits in 32 bits (m <= 32) for 2x the lane
// count. Reached generically via Avx512Ops::U32 — the kernels stay
// free of raw intrinsics and #ifdefs; the arch code lives here in the
// Ops layer like every other backend primitive. Bitwise ops are
// width-agnostic (same _si512 intrinsic); only set1/add/sub and the
// PEQ gather differ from the 64-bit struct. The gather is scalar L1
// lookups assembled into a register (the microcoded _mm512_i32gather
// was ~2x slower per pair on avx10_512).
struct Avx512OpsU32 {
  static constexpr std::size_t lanes = 16;
  using Vec = __m512i;
  using Lane = std::uint32_t;

  static Vec set1(std::uint32_t v) { return _mm512_set1_epi32(static_cast<int>(v)); }
  static Vec zero() { return _mm512_setzero_si512(); }
  static Vec and_(Vec a, Vec b) { return _mm512_and_si512(a, b); }
  static Vec or_(Vec a, Vec b) { return _mm512_or_si512(a, b); }
  static Vec xor_(Vec a, Vec b) { return _mm512_xor_si512(a, b); }
  static Vec not_(Vec a) { return _mm512_xor_si512(a, _mm512_set1_epi32(-1)); }
  static Vec add(Vec a, Vec b) { return _mm512_add_epi32(a, b); }
  static Vec sub(Vec a, Vec b) { return _mm512_sub_epi32(a, b); }
  static Vec gather(const std::uint32_t* base, const std::uint32_t* indices) {
    alignas(64) std::uint32_t tmp[lanes];
    for (std::size_t l = 0; l < lanes; ++l) {
      tmp[l] = base[indices[l]];
    }
    return _mm512_load_si512(reinterpret_cast<const void*>(tmp));
  }
  static void store_aligned(std::uint32_t* dst, Vec v) {
    _mm512_store_si512(reinterpret_cast<void*>(dst), v);
  }
  // Additional primitives for the Myers / OSA score recurrences. The
  // u32 kernels expand AVX-512's compare masks back to all-ones/zero
  // vectors (as the 64-bit struct does) so the kernel logic stays
  // vector-only and identical across backends.
  static Vec load_aligned(const std::uint32_t* data) {
    return _mm512_load_si512(reinterpret_cast<const void*>(data));
  }
  static Vec load_unaligned(const std::uint32_t* data) {
    return _mm512_loadu_si512(reinterpret_cast<const void*>(data));
  }
  static Vec andnot_(Vec a, Vec b) { return _mm512_andnot_si512(a, b); }
  static Vec shl1(Vec a) { return _mm512_add_epi32(a, a); }
  static Vec gt(Vec a, Vec b) {
    return _mm512_maskz_set1_epi32(_mm512_cmpgt_epu32_mask(a, b), -1);
  }
  static Vec cmpeq(Vec a, Vec b) {
    return _mm512_maskz_set1_epi32(_mm512_cmpeq_epi32_mask(a, b), -1);
  }
};

// 32-lane uint16 / 64-lane uint8 siblings, reached via Avx512Ops::U16 /
// ::U8 for the packed-PEQ Indel/LCS cdist flip when the longest choice
// fits 16 / 8 bits. The Allison-Dix recurrence (U = V & PEQ;
// V = (V+U) | (V-U)) only needs and/or/add/sub: the add's carry-out of
// the top lane bit is discarded exactly as it is for the 64-bit struct,
// so a len<=W choice is bit-identical at any lane width W. Packing into
// the narrowest lane that fits doubles/quadruples the lane count (32/64
// vs 16) for the common short-string case — matching rapidfuzz's
// element-width dispatch. Needs AVX-512BW for the epi16/epi8 add/sub.
struct Avx512OpsU16 {
  static constexpr std::size_t lanes = 32;
  using Vec = __m512i;
  using Lane = std::uint16_t;

  static Vec set1(std::uint16_t v) { return _mm512_set1_epi16(static_cast<short>(v)); }
  static Vec zero() { return _mm512_setzero_si512(); }
  static Vec and_(Vec a, Vec b) { return _mm512_and_si512(a, b); }
  static Vec or_(Vec a, Vec b) { return _mm512_or_si512(a, b); }
  static Vec add(Vec a, Vec b) { return _mm512_add_epi16(a, b); }
  static Vec sub(Vec a, Vec b) { return _mm512_sub_epi16(a, b); }
  static Vec load_aligned(const std::uint16_t* d) {
    return _mm512_load_si512(reinterpret_cast<const void*>(d));
  }
  static Vec load_unaligned(const std::uint16_t* d) {
    return _mm512_loadu_si512(reinterpret_cast<const void*>(d));
  }
  static void store_aligned(std::uint16_t* dst, Vec v) {
    _mm512_store_si512(reinterpret_cast<void*>(dst), v);
  }
};

struct Avx512OpsU8 {
  static constexpr std::size_t lanes = 64;
  using Vec = __m512i;
  using Lane = std::uint8_t;

  static Vec set1(std::uint8_t v) { return _mm512_set1_epi8(static_cast<char>(v)); }
  static Vec zero() { return _mm512_setzero_si512(); }
  static Vec and_(Vec a, Vec b) { return _mm512_and_si512(a, b); }
  static Vec or_(Vec a, Vec b) { return _mm512_or_si512(a, b); }
  static Vec add(Vec a, Vec b) { return _mm512_add_epi8(a, b); }
  static Vec sub(Vec a, Vec b) { return _mm512_sub_epi8(a, b); }
  static Vec load_aligned(const std::uint8_t* d) {
    return _mm512_load_si512(reinterpret_cast<const void*>(d));
  }
  static Vec load_unaligned(const std::uint8_t* d) {
    return _mm512_loadu_si512(reinterpret_cast<const void*>(d));
  }
  static void store_aligned(std::uint8_t* dst, Vec v) {
    _mm512_store_si512(reinterpret_cast<void*>(dst), v);
  }
};

struct Avx512Ops {
  static constexpr std::size_t lanes = 8;
  using Vec = __m512i;
  using Lane = std::uint64_t;
  using U32 = Avx512OpsU32;
  using U16 = Avx512OpsU16;
  using U8 = Avx512OpsU8;
  // Fallback transpose Ops for the packed-PEQ cdist flip when the
  // longest choice needs the full 64 bits. The Indel flip dispatches on
  // the max choice length and prefers the narrowest of U8/U16/U32 that
  // fits (more lanes per 512-bit op), falling back to this 64-bit/8-lane
  // self only for choices in (32, 64]. (The old scattered-gather path
  // preferred the 8-lane self to dodge a store-forwarding stall; the
  // flip's loads are contiguous, so lane count wins instead.)
  using CdistTransposeOps = Avx512Ops;

  static Vec set1(std::uint64_t v) {
    return _mm512_set1_epi64(static_cast<long long>(v));
  }
  static Vec zero() { return _mm512_setzero_si512(); }
  static Vec and_(Vec a, Vec b) { return _mm512_and_si512(a, b); }
  static Vec or_(Vec a, Vec b) { return _mm512_or_si512(a, b); }
  static Vec xor_(Vec a, Vec b) { return _mm512_xor_si512(a, b); }
  static Vec not_(Vec a) {
    return _mm512_xor_si512(a, _mm512_set1_epi64(static_cast<long long>(-1LL)));
  }
  static Vec add(Vec a, Vec b) { return _mm512_add_epi64(a, b); }
  static Vec sub(Vec a, Vec b) { return _mm512_sub_epi64(a, b); }
  static Vec shl1(Vec a) { return _mm512_add_epi64(a, a); }
  static Vec cmpeq(Vec a, Vec b) {
    // AVX-512 produces a mask; expand back to a vector of all-ones / zeros.
    return _mm512_maskz_set1_epi64(_mm512_cmpeq_epi64_mask(a, b), -1LL);
  }
  static Vec andnot_(Vec a, Vec b) { return _mm512_andnot_si512(a, b); }
  static Vec gather64(const std::uint64_t* base, const std::uint64_t* indices) {
    const __m512i idx = _mm512_load_si512(reinterpret_cast<const void*>(indices));
    return _mm512_i64gather_epi64(idx, reinterpret_cast<const void*>(base), 8);
  }
  static Vec load_aligned(const std::uint64_t* data) {
    return _mm512_load_si512(reinterpret_cast<const void*>(data));
  }
  static Vec load_unaligned(const std::uint64_t* data) {
    return _mm512_loadu_si512(reinterpret_cast<const void*>(data));
  }
  static void store_aligned(std::uint64_t* dst, Vec v) {
    _mm512_store_si512(reinterpret_cast<void*>(dst), v);
  }
  static Vec gt_u64(Vec a, Vec b) {
    // AVX-512 has a true unsigned variant.
    return _mm512_maskz_set1_epi64(_mm512_cmpgt_epu64_mask(a, b), -1LL);
  }
  static Vec shr63(Vec a) { return _mm512_srli_epi64(a, 63); }
  static bool is_zero(Vec a) { return _mm512_test_epi64_mask(a, a) == 0; }
  static Vec shl_var_u64(Vec a, Vec shifts) {
    return _mm512_sllv_epi64(a, shifts);
  }
  static Vec shr_var_u64(Vec a, Vec shifts) {
    return _mm512_srlv_epi64(a, shifts);
  }
};

#endif  // __AVX512F__

#if defined(__ARM_NEON)

// 4-lane uint32 NEON sibling of NeonOps (uint32x4_t), reached via
// NeonOps::U32 for the bit-parallel batch kernels when the query fits
// in 32 bits (m <= 32) — 2x the lane count of the 64-bit struct. NEON
// has native unsigned cmpgt_u32 and a real vmvnq_u32 NOT, so the u32
// struct is actually a touch simpler than its 64-bit sibling. No
// native gather: four scalar lane inserts.
struct NeonOpsU32 {
  static constexpr std::size_t lanes = 4;
  using Vec = uint32x4_t;
  using Lane = std::uint32_t;

  static Vec set1(std::uint32_t v) { return vdupq_n_u32(v); }
  static Vec zero() { return vdupq_n_u32(0); }
  static Vec and_(Vec a, Vec b) { return vandq_u32(a, b); }
  static Vec or_(Vec a, Vec b) { return vorrq_u32(a, b); }
  static Vec xor_(Vec a, Vec b) { return veorq_u32(a, b); }
  static Vec not_(Vec a) { return vmvnq_u32(a); }
  static Vec add(Vec a, Vec b) { return vaddq_u32(a, b); }
  static Vec sub(Vec a, Vec b) { return vsubq_u32(a, b); }
  static Vec shl1(Vec a) { return vshlq_n_u32(a, 1); }
  static Vec gt(Vec a, Vec b) { return vcgtq_u32(a, b); }
  static Vec cmpeq(Vec a, Vec b) { return vceqq_u32(a, b); }
  // ~a & b. vbicq computes a & ~b, so swap operands.
  static Vec andnot_(Vec a, Vec b) { return vbicq_u32(b, a); }
  static Vec gather(const std::uint32_t* base, const std::uint32_t* indices) {
    Vec out = vdupq_n_u32(base[indices[0]]);
    out = vsetq_lane_u32(base[indices[1]], out, 1);
    out = vsetq_lane_u32(base[indices[2]], out, 2);
    out = vsetq_lane_u32(base[indices[3]], out, 3);
    return out;
  }
  static Vec load_aligned(const std::uint32_t* data) { return vld1q_u32(data); }
  // NEON vld1q has no alignment requirement, so unaligned == aligned.
  static Vec load_unaligned(const std::uint32_t* data) { return vld1q_u32(data); }
  static void store_aligned(std::uint32_t* dst, Vec v) { vst1q_u32(dst, v); }
};

// 128-bit NEON: 2 lanes of 64-bit. Same lane count as SseOps; the
// difference is NEON's native unsigned cmpgt_u64 (no XOR-sign trick
// needed) and the slightly weirder ~a (uint32 lane reinterpret).
// Used by the macos_arm64_neon backend (Apple Silicon M-series) and
// the linux_aarch64_neon backend (Graviton4 etc.).
struct NeonOps {
  static constexpr std::size_t lanes = 2;
  using Vec = uint64x2_t;
  using Lane = std::uint64_t;
  using U32 = NeonOpsU32;
  // Best Ops for the pre-transposed cdist path. NEON inserts lanes via
  // register moves (vsetq_lane, no memory round-trip), so there's no
  // store-forwarding penalty — the 32-bit/4-lane sibling's extra
  // parallelism wins outright.
  using CdistTransposeOps = NeonOpsU32;

  static Vec set1(std::uint64_t v) { return vdupq_n_u64(v); }
  static Vec zero() { return vdupq_n_u64(0); }
  static Vec and_(Vec a, Vec b) { return vandq_u64(a, b); }
  static Vec or_(Vec a, Vec b) { return vorrq_u64(a, b); }
  static Vec xor_(Vec a, Vec b) { return veorq_u64(a, b); }
  static Vec not_(Vec a) {
    // NEON has vmvnq_u32 but no 64-bit equivalent — reinterpret through
    // u32 and back.
    return vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(a)));
  }
  static Vec add(Vec a, Vec b) { return vaddq_u64(a, b); }
  static Vec sub(Vec a, Vec b) { return vsubq_u64(a, b); }
  static Vec shl1(Vec a) { return vshlq_n_u64(a, 1); }
  static Vec cmpeq(Vec a, Vec b) { return vceqq_u64(a, b); }
  // ~a & b. NEON's `vbicq` computes `a & ~b` (bit clear), so we have
  // to swap the operands.
  static Vec andnot_(Vec a, Vec b) { return vbicq_u64(b, a); }
  static Vec gather64(const std::uint64_t* base, const std::uint64_t* indices) {
    // No native NEON gather — issue two scalar loads.
    Vec out = vdupq_n_u64(base[indices[0]]);
    out = vsetq_lane_u64(base[indices[1]], out, 1);
    return out;
  }
  static Vec load_aligned(const std::uint64_t* data) { return vld1q_u64(data); }
  static void store_aligned(std::uint64_t* dst, Vec v) { vst1q_u64(dst, v); }
  // NEON has native unsigned cmpgt for 64-bit lanes (ARMv8+).
  static Vec gt_u64(Vec a, Vec b) { return vcgtq_u64(a, b); }
  static Vec shr63(Vec a) { return vshrq_n_u64(a, 63); }
  static bool is_zero(Vec a) {
    // OR both halves; result is zero iff every bit of v is zero.
    uint64x1_t combined = vorr_u64(vget_low_u64(a), vget_high_u64(a));
    return vget_lane_u64(combined, 0) == 0;
  }
  // NEON's vshlq_u64 takes a SIGNED shift count: positive shifts left,
  // negative shifts right. We expose left/right as two helpers using
  // the same underlying instruction.
  static Vec shl_var_u64(Vec a, Vec shifts) {
    return vshlq_u64(a, vreinterpretq_s64_u64(shifts));
  }
  static Vec shr_var_u64(Vec a, Vec shifts) {
    return vshlq_u64(a, vnegq_s64(vreinterpretq_s64_u64(shifts)));
  }
};

#endif  // __ARM_NEON

#if defined(__loongarch_sx)

// LoongArch LSX: 128-bit, 2x 64-bit lanes. Same lane count as SSE/NEON.
// LSX semantics quirks:
//   * vandn_v(a, b) = a & ~b (operand order swapped vs Intel andnot)
//   * NOT via NOR with self
//   * No native gather — scalar lane inserts
//   * Unsigned cmpgt via vslt_du with operands swapped
// Used by the linux_loongarch64_lsx backend.
// 4-lane uint32 sibling of LsxOps (128-bit / 32), reached via
// LsxOps::U32 for the bit-parallel batch + packed-PEQ cdist paths.
struct LsxOpsU32 {
  static constexpr std::size_t lanes = 4;
  using Vec = __m128i;
  using Lane = std::uint32_t;

  static Vec set1(std::uint32_t v) {
    return __lsx_vreplgr2vr_w(static_cast<int>(v));
  }
  static Vec zero() { return __lsx_vreplgr2vr_w(0); }
  static Vec and_(Vec a, Vec b) { return __lsx_vand_v(a, b); }
  static Vec or_(Vec a, Vec b) { return __lsx_vor_v(a, b); }
  static Vec xor_(Vec a, Vec b) { return __lsx_vxor_v(a, b); }
  static Vec not_(Vec a) { return __lsx_vnor_v(a, a); }
  static Vec add(Vec a, Vec b) { return __lsx_vadd_w(a, b); }
  static Vec sub(Vec a, Vec b) { return __lsx_vsub_w(a, b); }
  static Vec shl1(Vec a) { return __lsx_vslli_w(a, 1); }
  static Vec cmpeq(Vec a, Vec b) { return __lsx_vseq_w(a, b); }
  static Vec andnot_(Vec a, Vec b) { return __lsx_vandn_v(a, b); }
  static Vec gt(Vec a, Vec b) { return __lsx_vslt_wu(b, a); }
  static Vec gather(const std::uint32_t* base, const std::uint32_t* idx) {
    Vec out = __lsx_vreplgr2vr_w(static_cast<int>(base[idx[0]]));
    out = __lsx_vinsgr2vr_w(out, static_cast<int>(base[idx[1]]), 1);
    out = __lsx_vinsgr2vr_w(out, static_cast<int>(base[idx[2]]), 2);
    out = __lsx_vinsgr2vr_w(out, static_cast<int>(base[idx[3]]), 3);
    return out;
  }
  static Vec load_aligned(const std::uint32_t* data) {
    return __lsx_vld(const_cast<void*>(reinterpret_cast<const void*>(data)), 0);
  }
  static Vec load_unaligned(const std::uint32_t* data) {
    return __lsx_vld(const_cast<void*>(reinterpret_cast<const void*>(data)), 0);
  }
  static void store_aligned(std::uint32_t* dst, Vec v) {
    __lsx_vst(v, reinterpret_cast<void*>(dst), 0);
  }
};

struct LsxOps {
  static constexpr std::size_t lanes = 2;
  using Vec = __m128i;
  using Lane = std::uint64_t;
  using U32 = LsxOpsU32;
  using CdistTransposeOps = LsxOpsU32;

  static Vec set1(std::uint64_t v) {
    return __lsx_vreplgr2vr_d(static_cast<long>(v));
  }
  static Vec zero() { return __lsx_vreplgr2vr_d(0); }
  static Vec and_(Vec a, Vec b) { return __lsx_vand_v(a, b); }
  static Vec or_(Vec a, Vec b) { return __lsx_vor_v(a, b); }
  static Vec xor_(Vec a, Vec b) { return __lsx_vxor_v(a, b); }
  static Vec not_(Vec a) { return __lsx_vnor_v(a, a); }
  static Vec add(Vec a, Vec b) { return __lsx_vadd_d(a, b); }
  static Vec sub(Vec a, Vec b) { return __lsx_vsub_d(a, b); }
  static Vec shl1(Vec a) { return __lsx_vslli_d(a, 1); }
  static Vec cmpeq(Vec a, Vec b) { return __lsx_vseq_d(a, b); }
  static Vec andnot_(Vec a, Vec b) {
    // LSX vandn(x, y) = ~x & y (Intel-style), contrary to what the
    // LoongArch ISA reference's "AND-NOT" mnemonic suggests when read
    // against Power's vandc or ARM's bic (both of which are a & ~b).
    // The ISA pseudocode states the formula correctly; the mnemonic
    // is the trap. See docs/loongarch_lsx_lasx_vandn_gotcha.md for the
    // full story.
    return __lsx_vandn_v(a, b);
  }
  static Vec gather64(const std::uint64_t* base, const std::uint64_t* indices) {
    Vec out = __lsx_vreplgr2vr_d(static_cast<long>(base[indices[0]]));
    out = __lsx_vinsgr2vr_d(out, static_cast<long>(base[indices[1]]), 1);
    return out;
  }
  static Vec load_aligned(const std::uint64_t* data) {
    return __lsx_vld(const_cast<void*>(reinterpret_cast<const void*>(data)), 0);
  }
  static void store_aligned(std::uint64_t* dst, Vec v) {
    __lsx_vst(v, reinterpret_cast<void*>(dst), 0);
  }
  // Unsigned cmpgt: vslt_du(b, a) returns a > b unsigned.
  static Vec gt_u64(Vec a, Vec b) { return __lsx_vslt_du(b, a); }
  static Vec shr63(Vec a) { return __lsx_vsrli_d(a, 63); }
  static bool is_zero(Vec a) {
    const long lo = __lsx_vpickve2gr_du(a, 0);
    const long hi = __lsx_vpickve2gr_du(a, 1);
    return (lo | hi) == 0;
  }
  static Vec shl_var_u64(Vec a, Vec shifts) { return __lsx_vsll_d(a, shifts); }
  static Vec shr_var_u64(Vec a, Vec shifts) { return __lsx_vsrl_d(a, shifts); }
};

#endif  // __loongarch_sx

#if defined(__loongarch_asx)

// 8-lane uint32 sibling of LasxOps (256-bit / 32), reached via
// LasxOps::U32 for the bit-parallel batch + packed-PEQ cdist paths when
// the pattern fits 32 bits — 2x the lane count of the 64-bit struct.
// `_w` (word) intrinsics replace the `_d` (doubleword) ones; bitwise ops
// are width-agnostic.
struct LasxOpsU32 {
  static constexpr std::size_t lanes = 8;
  using Vec = __m256i;
  using Lane = std::uint32_t;

  static Vec set1(std::uint32_t v) {
    return __lasx_xvreplgr2vr_w(static_cast<int>(v));
  }
  static Vec zero() { return __lasx_xvreplgr2vr_w(0); }
  static Vec and_(Vec a, Vec b) { return __lasx_xvand_v(a, b); }
  static Vec or_(Vec a, Vec b) { return __lasx_xvor_v(a, b); }
  static Vec xor_(Vec a, Vec b) { return __lasx_xvxor_v(a, b); }
  static Vec not_(Vec a) { return __lasx_xvnor_v(a, a); }
  static Vec add(Vec a, Vec b) { return __lasx_xvadd_w(a, b); }
  static Vec sub(Vec a, Vec b) { return __lasx_xvsub_w(a, b); }
  static Vec shl1(Vec a) { return __lasx_xvslli_w(a, 1); }
  static Vec cmpeq(Vec a, Vec b) { return __lasx_xvseq_w(a, b); }
  static Vec andnot_(Vec a, Vec b) { return __lasx_xvandn_v(a, b); }
  static Vec gt(Vec a, Vec b) { return __lasx_xvslt_wu(b, a); }
  static Vec gather(const std::uint32_t* base, const std::uint32_t* idx) {
    Vec out = __lasx_xvreplgr2vr_w(static_cast<int>(base[idx[0]]));
    out = __lasx_xvinsgr2vr_w(out, static_cast<int>(base[idx[1]]), 1);
    out = __lasx_xvinsgr2vr_w(out, static_cast<int>(base[idx[2]]), 2);
    out = __lasx_xvinsgr2vr_w(out, static_cast<int>(base[idx[3]]), 3);
    out = __lasx_xvinsgr2vr_w(out, static_cast<int>(base[idx[4]]), 4);
    out = __lasx_xvinsgr2vr_w(out, static_cast<int>(base[idx[5]]), 5);
    out = __lasx_xvinsgr2vr_w(out, static_cast<int>(base[idx[6]]), 6);
    out = __lasx_xvinsgr2vr_w(out, static_cast<int>(base[idx[7]]), 7);
    return out;
  }
  static Vec load_aligned(const std::uint32_t* data) {
    return __lasx_xvld(const_cast<void*>(reinterpret_cast<const void*>(data)), 0);
  }
  // LoongArch vector loads tolerate unaligned addresses.
  static Vec load_unaligned(const std::uint32_t* data) {
    return __lasx_xvld(const_cast<void*>(reinterpret_cast<const void*>(data)), 0);
  }
  static void store_aligned(std::uint32_t* dst, Vec v) {
    __lasx_xvst(v, reinterpret_cast<void*>(dst), 0);
  }
};

// LoongArch LASX: 256-bit, 4x 64-bit lanes. Same lane count as AVX2.
// Same semantics as LSX but with `xv*` intrinsics.
struct LasxOps {
  static constexpr std::size_t lanes = 4;
  using Vec = __m256i;
  using Lane = std::uint64_t;
  using U32 = LasxOpsU32;
  // The packed-PEQ cdist path loads contiguous PEQ columns, so more
  // lanes wins (no store-forwarding penalty); use the 8-lane sibling.
  using CdistTransposeOps = LasxOpsU32;

  static Vec set1(std::uint64_t v) {
    return __lasx_xvreplgr2vr_d(static_cast<long>(v));
  }
  static Vec zero() { return __lasx_xvreplgr2vr_d(0); }
  static Vec and_(Vec a, Vec b) { return __lasx_xvand_v(a, b); }
  static Vec or_(Vec a, Vec b) { return __lasx_xvor_v(a, b); }
  static Vec xor_(Vec a, Vec b) { return __lasx_xvxor_v(a, b); }
  static Vec not_(Vec a) { return __lasx_xvnor_v(a, a); }
  static Vec add(Vec a, Vec b) { return __lasx_xvadd_d(a, b); }
  static Vec sub(Vec a, Vec b) { return __lasx_xvsub_d(a, b); }
  static Vec shl1(Vec a) { return __lasx_xvslli_d(a, 1); }
  static Vec cmpeq(Vec a, Vec b) { return __lasx_xvseq_d(a, b); }
  static Vec andnot_(Vec a, Vec b) {
    // Same Intel-style convention as LSX: xvandn(x, y) = ~x & y.
    // See docs/loongarch_lsx_lasx_vandn_gotcha.md.
    return __lasx_xvandn_v(a, b);
  }
  static Vec gather64(const std::uint64_t* base, const std::uint64_t* indices) {
    Vec out = __lasx_xvreplgr2vr_d(static_cast<long>(base[indices[0]]));
    out = __lasx_xvinsgr2vr_d(out, static_cast<long>(base[indices[1]]), 1);
    out = __lasx_xvinsgr2vr_d(out, static_cast<long>(base[indices[2]]), 2);
    out = __lasx_xvinsgr2vr_d(out, static_cast<long>(base[indices[3]]), 3);
    return out;
  }
  static Vec load_aligned(const std::uint64_t* data) {
    return __lasx_xvld(const_cast<void*>(reinterpret_cast<const void*>(data)), 0);
  }
  static void store_aligned(std::uint64_t* dst, Vec v) {
    __lasx_xvst(v, reinterpret_cast<void*>(dst), 0);
  }
  static Vec gt_u64(Vec a, Vec b) { return __lasx_xvslt_du(b, a); }
  static Vec shr63(Vec a) { return __lasx_xvsrli_d(a, 63); }
  static bool is_zero(Vec a) {
    const long l0 = __lasx_xvpickve2gr_du(a, 0);
    const long l1 = __lasx_xvpickve2gr_du(a, 1);
    const long l2 = __lasx_xvpickve2gr_du(a, 2);
    const long l3 = __lasx_xvpickve2gr_du(a, 3);
    return (l0 | l1 | l2 | l3) == 0;
  }
  static Vec shl_var_u64(Vec a, Vec shifts) { return __lasx_xvsll_d(a, shifts); }
  static Vec shr_var_u64(Vec a, Vec shifts) { return __lasx_xvsrl_d(a, shifts); }
};

#endif  // __loongarch_asx

#if defined(__VSX__) || defined(__POWER8_VECTOR__)

// PowerPC VSX: 128-bit, 2x 64-bit lanes. Similar shape to SSE / NEON /
// LSX. Used by the linux_powerpc64_vsx backend (Power8+, ISA 2.07).
// Conventions worth noting:
//   * vec_andc(a, b) = a & ~b (NOT Intel-style), so andnot_ swaps
//     operands to get ~a & b.
//   * NOT via vec_nor with self.
//   * No native gather — manual lane inserts via array-style indexing.
//   * vec_cmpgt for unsigned long long is ISA 2.07 native.
//   * Shift counts are per-lane vectors, not scalars (vec_sl/vec_sr).
struct VsxOps {
  static constexpr std::size_t lanes = 2;
  using Vec = __vector unsigned long long;

  static Vec set1(std::uint64_t v) {
    return vec_splats(static_cast<unsigned long long>(v));
  }
  static Vec zero() { return vec_splats(0ULL); }
  static Vec and_(Vec a, Vec b) { return vec_and(a, b); }
  static Vec or_(Vec a, Vec b) { return vec_or(a, b); }
  static Vec xor_(Vec a, Vec b) { return vec_xor(a, b); }
  static Vec not_(Vec a) { return vec_nor(a, a); }
  static Vec add(Vec a, Vec b) { return vec_add(a, b); }
  static Vec sub(Vec a, Vec b) { return vec_sub(a, b); }
  // vec_sl/vec_sr take per-lane shift counts.
  static Vec shl1(Vec a) { return vec_sl(a, vec_splats(1ULL)); }
  static Vec cmpeq(Vec a, Vec b) {
    return reinterpret_cast<Vec>(vec_cmpeq(a, b));
  }
  // ~a & b. Power's vec_andc(x, y) = x & ~y, so swap operands.
  static Vec andnot_(Vec a, Vec b) { return vec_andc(b, a); }
  static Vec gather64(const std::uint64_t* base, const std::uint64_t* indices) {
    Vec out = {base[indices[0]], base[indices[1]]};
    return out;
  }
  static Vec load_aligned(const std::uint64_t* data) {
    // Use vec_xl rather than `*reinterpret_cast<const Vec*>(data)` to
    // sidestep strict-aliasing reordering: GCC at -O3 is free to move
    // scalar writes to `data` past a same-block Vec read through a
    // reinterpret_cast (the Vec type is distinct from uint64_t), and
    // the Jaro batch kernel's per-iteration scratch pattern is exactly
    // the shape that triggers it. vec_xl is an explicit load
    // intrinsic so the compiler must honor write-before-load ordering.
    return vec_xl(0, const_cast<unsigned long long*>(
        reinterpret_cast<const unsigned long long*>(data)));
  }
  static void store_aligned(std::uint64_t* dst, Vec v) {
    vec_xst(v, 0, reinterpret_cast<unsigned long long*>(dst));
  }
  // Native unsigned 64-bit cmpgt on Power8+ (ISA 2.07).
  static Vec gt_u64(Vec a, Vec b) {
    return reinterpret_cast<Vec>(vec_cmpgt(a, b));
  }
  static Vec shr63(Vec a) { return vec_sr(a, vec_splats(63ULL)); }
  static bool is_zero(Vec a) {
    return (vec_extract(a, 0) | vec_extract(a, 1)) == 0ULL;
  }
  // Power's vec_sl / vec_sr take per-lane unsigned shift counts.
  static Vec shl_var_u64(Vec a, Vec shifts) { return vec_sl(a, shifts); }
  static Vec shr_var_u64(Vec a, Vec shifts) { return vec_sr(a, shifts); }
};

#endif  // __VSX__

}  // namespace stride_align::levenshtein_simd
