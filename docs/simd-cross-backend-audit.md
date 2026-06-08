# Cross-backend SIMD Audit — Short-String Local Smith-Waterman (priority: sub-100-char, match/mismatch, score-only)

## Cross-cutting

These dominate the audit. The first three are the highest-leverage and recur on every backend.

### 1. Local-SW width over-selection — `compute_score_bound` is alignment-mode-agnostic (HIGH, high) — **confirmed**
`compute_score_bound` (`preprocess.hpp:124-141`) returns `(query+target)*max|step|` with no local/global distinction (verified). Local SW clamps every cell at 0 (`farrar_fixed_kernel.hpp:3564`, `:3314`; scalable `:1247`), so the true reachable max is `(min(q,t)+1)*match`, not `(q+t)*max|step|`. `select_score_bits` (`farrar_preprocess.hpp:70`) consumes only the bound, and the width is computed **once** and shared by SW+NW (`farrar_preprocess.hpp:1132-1141`). Canonical 60×60 match=2: bound 240 → int16, true local max 122 → fits int8. This single mis-selection halves lane density on the priority workload across **every** backend:

| Backend | int8→int16 lane loss |
|---|---|
| SSE4.1 / NEON / LSX | 16 → 8 |
| AVX2 / LASX | 32 → 16 |
| AVX512 / AVX10-512 | 64 → 32 |
| RVV (VLEN=128) | 16 → 8 (full register width halved) |
| SVE (128b) | 16 → 8 |

**Fix:** thread an `is_local` flag from the SW dispatch into a local-only bound `min(q,t) * max(magnitude(match), mismatch>0 ? magnitude(mismatch) : 0)`; keep `(q+t)*max|step|` for NW and fall back to it when any gap score >0 (positive gaps can raise a clamped cell via lazy-F, `track_lazy_best` at `farrar_fixed_kernel.hpp:3538`). Pure preprocess change, no kernel edits, zero correctness risk in the standard SW regime. **This is the precondition that lets every int8 fast-path below ever run; do it first.**
- **Caveat (from verifier):** the local bound still exceeds 127 once `min(q,t)*match > 126`, so the win is the genuinely-short / low-match band (the 60×60 example is in-band); and the bound expression *must* include `mismatch` when `mismatch>0` or it under-bounds and can pick a too-narrow cell. With the priority `mismatch≤0` the match-only form is safe.

### 2. `reduce_max` horizontal reduction falls to a scalar stack-spill loop (HIGH, high) — **confirmed, with correction**
The shared `reduce_max<Ops,Cell>` (`farrar_fixed_kernel.hpp:495-508`) uses `Ops::reduce_max` only when present; otherwise stores `best_vector` to an `alignas` stack array and runs a scalar `std::max` loop over `lane_count` (16/32 lanes for int8). This is the **final step of every score-only call** (`:3465`, `:3659`) on the priority path. Verified by member-count probe:

| Backend | has `reduce_max` member? |
|---|---|
| SSE4.1, AVX10-256, AVX10-512, NEON, LASX, LSX | **No** → scalar spill |
| AVX2 | **No member** (the 7 hits are free fns `reduce_max_i16/i32_256` + call sites, used only inside exact-fill kernels short strings never reach) → scalar spill |
| AVX512bwvl, **VSX** | Yes (members) |
| RVV (scalable kernel) | No → heap-allocating fallback (worse, see §6) |

**Correction to the shared finding:** it listed VSX as lacking `reduce_max` (citing the dead `powerpc_vsx_kernel.hpp`); the live `linux_powerpc64_vsx.hpp` **does** define it for all widths. Affected fixed backends are **SSE/AVX2/NEON/LSX/LASX/AVX10-256/AVX10-512** — not VSX. Also drop the `phminposuw` suggestion for int8 max-reduction (it is unsigned-16 *min*, wrong semantics).
**Fix per ISA (all on the priority path, all confirmed-compiling):** NEON `vmaxvq_s8/s16/s32` (single instr); SSE `_mm_max_epi8/16/32` + `_mm_shuffle/_mm_srli_si128` tree (int16 alt: `0x7FFF - phminposuw(0x7FFF-v)`); AVX2 `_mm256_extracti128_si256` + 128-bit fold (do **not** rely on the generic shift log-tree here — its lane-crossing `shift_left_zero_256` is 4 ops/step); LASX/LSX the in-file `xvmax/vmax + xvbsrl/vbsrl` log-tree already proven in their exact-fill kernels; AVX10-256/512 `_mm256_max_epi*`/SSE tail tree; RVV `vredmax.vs + vmv.x.s`. The portable shift+max log-tree in the shared fallback is correct only because `best_vector≥0` at all call sites.

### 3. Short-string `score_state` uses the naive O(lane_count) lazy-F loop, not the log-step prefix carry the exact-fill path already has (HIGH, high) — **confirmed**
Two local-SW lazy-F corrections coexist in `farrar_fixed_kernel.hpp`. The exact-fill path (`:3420-3462`) uses `local_lazy_f_prefix_carry` (O(log lane_count) shift+max) + one early-exiting bounded scan. The **generic** `score_state` (`:3584-3656`) runs the OLD naive loop: up to `lane_count` outer iterations, each reloading **all** `segment_count` h_store vectors. Exact-fill is gated to `query_size == segment_count*lane_count` with `segment_count ∈ {16…256}` (i.e. ≥256/1024-char queries), so **sub-100-char queries never reach it** and always take the naive loop. `local_lazy_f_prefix_carry` is keyed on `lane_count` (segment_count only scales `span_gap`), so it drops straight into `score_state`.
**Fix:** after the segment loop, replace the three naive tiers with `v_f = local_lazy_f_prefix_carry(...)` + `if (any_greater(v_f,zero)) scan_local_linear_lazy_f_once_bounded(...)`, mirroring exact-fill. Backends with the carry (SSE/AVX2/AVX512/LSX 8/16/32; NEON 8/16 only) get the full log path; LASX/VSX/NEON-32/64/AVX2-64 fall to the scalar carry but still gain the single bounded scan. The commit log attributes a 0.69×→competitive parasail swing to this exact change on the *global* path; local hasn't received it.
- **Magnitude caveat:** the naive loop already has a `gap≤0` early-exit and the F-wave typically propagates 1-2 lanes, so the worst-case `lane_count*segment_count` rarely materializes; the change is never slower and removes a data-dependent branch, but quantify the win on adversarial gap-run inputs rather than asserting it universally.

### 4. Per-call state allocation churn — `prepare_score_state` reallocates profile + h_store/h_load/e_store + offsets every call (MEDIUM, high) — **confirmed**
`score()` default-constructs a fresh `PreparedScoreState` per call (`farrar_fixed_kernel.hpp:873`, resizes at `:952-954`); the `*_scores`/cdist batch loop re-prepares per pair. For sub-100-char DPs the malloc/free + zero-fills are a large fraction of latency. The **matrix path already solved this** with a `thread_local PreparedScoreState` + in-place `assign()`/`clear()` (`:3806-3856`), and the affine score paths already use thread_local (`:2395` etc.) — the linear match/mismatch score path is the outlier. Generic scalar backend is worst-hit: 2N malloc/free for N targets, no profile/arena reuse (`generic.hpp:72-73`). Scalable kernel (SVE/RVV) re-mallocs profile+3 buffers+offsets per prepare too (`farrar_scalable_kernel.hpp:350-358`).
**Fix:** mirror the matrix path — `thread_local` keyed per `Cell`, in-place prepare, and for batch hoist the state above the per-target loop. Pure preprocess refactor, clear hot-path win.

### 5. Scalar profile build (`fill_score_profile_row`) (MEDIUM, medium) — **needs_benchmark**
`fill_score_profile_row` (`farrar_fixed_kernel.hpp:602-617`; scalable `:270-293`) writes one Cell per lane with a scalar compare — shared by every fixed and scalable backend. Each backend already has the vector primitive: `substitution()` is `cmpeq + blend(set1(match),set1(mismatch))`. Vectorizing requires a **precomputed striped-order query buffer** (the layout is `query[lane*segment_count+segment]`, strided) plus OOB-lane padding with an out-of-alphabet token. **The benefit scales with `lane_count*segment_count`, i.e. *longer* queries — not the sub-100-char priority case**, and the striped-query materialization may eat the savings for tiny queries. The stronger sub-claim is inline-substitution in the DP loop for high-cardinality short targets (skip the materialized profile entirely). Real and intrinsic-correct, but measure before committing. Shared-kernel work — not backend-specific.

### 6. `target_profile_offsets` is `std::vector<size_t>` (8B/char) (LOW, medium) — **mixed**
Stored value is `index*state_cells`, well under 2³² for all supported shapes; streamed once per DP column (`farrar_fixed_kernel.hpp:3545`, scalable `:1227`, affine `:1809`). uint32 halves it; folded uint16 index (×`state_cells` in-loop) quarters it. **For the priority sub-100-char workload the whole table is <800 B and L1-resident, so this is effectively a no-op (needs_benchmark / footprint-only)**; the real value is on long Chinese/UCS-2 targets. The cleanest concrete win is *dropping the table entirely* for the `target_ordered` layout (offsets are exactly `target_index*state_cells`). Note: a naive `size_t→uint32` change ripples into the `std::span<const std::size_t>` signatures on the exact-fill raw kernels — prefer the folded-index variant. Add a 64-bit fallback guard for pathological long×long wide-cell combos.

### Prior-finding reconciliation
1. **Width bound (local-agnostic):** CONFIRMED — see §1. Live and maximal on every backend; biggest single lever.
2. **8-bit unsigned-saturating + bias:** **needs_benchmark on every backend** (SSE `sse-uint8-saturating-bias`, AVX512 `int8-unsigned-saturating-repr`, AVX10-256, LASX, LSX, RVV). Intrinsics exist everywhere (`adds_epu8/max_epu8`, `vsadd_bu/vmax_bu`, `vsaddu`). It is a representation+dispatch redesign (bias through profile/readout/zero-clamp), pays off **only if §1 also lands**, and adds inner-loop ops — do **not** treat as a sure win. Ties to §7 below (fuses away the `max(.,zero)` clamp).
3. **Vectorize profile build:** CONFIRMED-as-real, needs_benchmark on speed — see §5.
4. **Inline substitution (skip profile):** folded into §5; promising for high-cardinality short targets.
5. **Per-call allocation churn:** CONFIRMED — see §4. Strongest of the "manifests-everywhere" items after §1-3.
6. **uint32 offsets:** CONFIRMED-real, marginal/footprint — see §6.

### 7. `max(v_h, zero)` clamp is fusable with the saturating representation (LOW, medium) — **needs_benchmark**
Inner loop issues three dependent maxes (`farrar_fixed_kernel.hpp:3562-3564`), the last purely to clamp ≥0. Under unsigned-saturating+bias (§2 / prior #2) the 0-floor is free via saturation. Only valid jointly with `adds_epu8`+`max_epu8`+bias (signed `adds_epi8` floors at −128, won't substitute); gate via an `Ops::saturating_local_floor` flag. Contingent on the §2 redesign landing.

---

## Per-backend

### x86-sse41
- **[HIGH, high] `x86_sse41.hpp` (no reduce_max member) — confirmed.** Add `reduce_max` per width; int16 = `_mm_extract_epi16(_mm_minpos_epu16(_mm_sub_epi16(set1(0x7FFF),v)),0)` complemented (~3 ops), int8 = `_mm_cvtepi8_epi16` widen both halves + minpos-complement, int32 = `_mm_max_epi32`+`_mm_shuffle_epi32` tree. Removes store-forwarding stall + 8-16 scalar compares from every call. (= §2)
- **[HIGH, high] `local_lazy_f_prefix_carry` redundant blendv — `x86_sse41.hpp:121-144/230-249/335-350` — confirmed.** `shift_left_insert_128<N>(prefix, zero_vector)` blends zero into bytes `_mm_slli_si128` already zeroed. Replace with `Ops::shift_left_zero` / direct `_mm_slli_si128`; drops one port-5 `_mm_blendv_epi8` + one mask load per log-step (4 int8/3 int16/2 int32 +1 final). On the hot exact-fill column path.
- **[MEDIUM, medium] No `global_lazy_f_prefix_carry` — `x86_sse41.hpp` absent; fallback `farrar_fixed_kernel.hpp:1578-1597` — confirmed.** NW prefix scan runs scalar (store/loop/reload). Add vectorized carry with `_mm_slli_si128` + `add_sentinel` (already defined `:103-107` etc). NW-only, off priority.
- **[MEDIUM, needs_benchmark] int64 `max` memory round-trip — `x86_sse41.hpp:422-432`.** Stores both operands, scalar ternary, reloads. **The finding's preferred fix (`_mm_cmpgt_epi64`) is SSE4.2, but the backend builds `-msse4.1` only — invalid on target.** A correct register-only SSE4.1 path exists (~9 ops: 2×`cmpgt_epi32`+`cmpeq_epi32`+xor-bias+and+or+shuffle+blendv) but the finding doesn't spell it out. 2-lane non-hot path; verify it actually beats the round-trip.
- **[MEDIUM, needs_benchmark] uint8 signed-non-saturating → bias — `x86_sse41.hpp:99-111`.** See §2; architectural, gated on §1.
- **[LOW, high] int64 substitution/add_sentinel — `x86_sse41.hpp:416-449` — confirmed optimal, do not touch.** `_mm_cmpeq_epi64`+`_mm_blendv_epi8` is correct; guard so the int64 `max` fix doesn't regress these.

### x86-avx2
- **[MEDIUM→effectively HIGH on priority, needs_benchmark] No reduce_max member — `x86_avx2.hpp:997-1503` — confirmed real, see §2.** The 7 hits are free fns used only in exact-fill kernels short strings can't reach (`segment_count` 32/64/128). Add member via `_mm256_extracti128_si256`+`_mm_max_epi*`+shuffle tree; int16 fold both halves with `_mm_max_epi16` *before* one `_mm_minpos_epu16`. Hot for **both int8 and int16** short strings. Benchmark only because a single reduction is tiny vs the DP body.
- **[LOW→downgraded, uncertain] int8 unbounded lazy-F — `farrar_fixed_kernel.hpp:3584`; flag `x86_avx2.hpp:1004`.** `bounded_local_sw_lazy_f_scan` is dead for the int8 generic path, but int8-eligible inputs cap near `q+t≤127` so `segment_count≤4` (not the cited 32), and the existing `any_nonzero` break at `:3602` already bounds iterations; at `gap==0` the bounded H+gap test reduces to the same condition → **~0 saving**. Note as dead capability, not a perf lever. (Subsumed by §3.)
- **[LOW, needs_benchmark] global NW `add_sentinel` → saturating add — `x86_avx2.hpp:1038-1042` etc; `:3739`.** For int8/int16 NW, represent `low_score` as signed-min and use `_mm256_adds_epi8/16` to pin −inf without `cmpeq+blendv`. Invasive sentinel-representation change across all NW call sites; NW-only.
- **[—] `shift_left_zero_256` 4-op cross-lane shift — `x86_avx2.hpp:41-47` — confirmed irreducible.** `permute2x128` required; `alignr_epi8` is per-128-lane and would drop the carry. Documentation only.
- **[—] int64 max `cmpgt_epi64+blendv` (`:1444`), aligned load/store (`:1014-1024`) — confirmed optimal/off-path.** No action.

### x86-avx512bwvl
- **[HIGH, high] width bound (`width-bound-local-aware`) — preprocess; payoff `x86_avx512bwvl.hpp:307/463` — confirmed.** = §1; 64→32 lane swing here.
- **[MEDIUM, high] int8/int16 reduce_max scalar tail — `x86_avx512bwvl.hpp:354-365 / 517-527` — confirmed.** Cross-128 fold is fine; finish the `__m128i` in-register (`_mm_max_epi8/16` + `_mm_unpackhi_epi64`/`_mm_shuffle_epi32 0xB1`/`_mm_srli_epi16(.,8)` then `_mm_extract`). Replaces 16B store + 15/7 scalar compares. Already more optimized than AVX2 here, so incremental.
- **[MEDIUM, needs_benchmark] no int8/int16 linear exact-fill `*_raw` — `x86_avx512bwvl.hpp:303-456`.** Only helps queries that exactly fill ≥16 segments (≥1024 int8 chars) — **off the sub-100-char priority path**; competes with the already-vectorized generic exact-fill, so delta is helper-call/regalloc only.
- **[MEDIUM, needs_benchmark] scalar profile build — see §5.** Shared, not avx512-specific.
- **[MEDIUM, needs_benchmark] int8 unsigned-saturating+bias — `:340-351` — see §2.**
- **[LOW, needs_benchmark] masked dense global scan unset — `:464-468`.** Add `store_masked_cells` (int32/64) + `masked_dense_global_lazy_f_scan` flag (int16 already has the store). NW-only; AVX-512 masked stores don't free a store slot, so benefit is marginal (avoided SLF hazards only). Inert on int8/int64 (no `dense_global_lazy_f_scan` there).

### x86-avx10-256
- **[HIGH, high] No reduce_max member — `x86_avx10_256.hpp:50-375` — confirmed.** int32 `castsi256_si128`+`extracti128`+`_mm_max_epi32`+`shuffle`+`cvtsi128_si32`; int64 can use **`_mm256_max_epi64` (vpmaxsq, available here via AVX512VL — unlike AVX2)**. See §2.
- **[HIGH, high] No `local_lazy_f_prefix_carry` — `x86_avx10_256.hpp:50-375`; generic mask loop `farrar_fixed_kernel.hpp:3606-3633` — confirmed.** Add the AVX2-style log-step (`shift_left_zero_256` + `set1(span_gap*2^k)` + `_mm256_max_epi*`/`_mm256_max_epi64`); O(log lane) vs up-to-32 passes.
- **[MEDIUM/HIGH, confirmed] No exact-fill / bounded / dense flags — `x86_avx10_256.hpp:50-375`.** Generic backing exists (`score_state_exact_fill_local_sw`, `scan_local_linear_lazy_f_once_bounded`); add `bounded_local_sw_lazy_f_scan`, `dense_global_lazy_f_scan`, `local_sw_score_exact_segmentN` flags. **Must be paired with the prefix-carry above** (exact-fill calls it at `:3420`) or it's a wash/regression. `bounded` alone is safe but smaller.
- **[MEDIUM, confirmed] No `global_lazy_f_prefix_carry` — `:50-375`; fallback `:1578-1597`.** Port AVX2 `global_lazy_f_prefix_carry_256`. NW-only.
- **[MEDIUM, needs_benchmark] int8 unsigned-saturating+bias — `:85-91` — see §2.**

### x86-avx10-512
- **[MEDIUM, high] No `local_lazy_f_prefix_carry` — `x86_avx10_512.hpp:56-382`; scalar fallback `farrar_fixed_kernel.hpp:1644-1664` — confirmed.** Copy the avx512bwvl carry (`:387-418` etc). The fallback genuinely stays scalar at -O3 (loop-carried dep, verified). **Not the sub-100-char hot path** (int8 segment_count=1 there); win is 100-4096-char and the UCS-2/4 int16/int32 paths. **Prereq:** port `shift_left_zero_bytes_512<16,32>` + `shift_left_insert_bytes_512` (see below).
- **[MEDIUM, high] No `global_lazy_f_prefix_carry` — `:56-382`; fallback `:1578-1597` — confirmed.** Port from avx512bwvl `:143-204` + `add_sentinel`. NW/affine-global only.
- **[MEDIUM, high] Missing exact-fill flags — `:56-382` — confirmed, narrow.** Long-query (≥1024 int8 / UCS-4 1024) only, off priority. **Correction: adding `bounded_local_sw_lazy_f_scan` alone is inert** — it's referenced only inside `score_state_exact_fill_local_sw` (`:3429`), which needs an exact-fill flag to enter. Add flags + raw kernel together.
- **[LOW, high] Missing `shift_left_zero_bytes_512<16,32>` / `shift_left_insert_bytes_512` — `:38-51` — confirmed.** Prerequisite enabler for the two carries above (`_mm512_maskz_permutexvar_epi64` idx `{0xFC}/{0xF0}`, `_mm512_mask_blend_epi8`). No standalone perf.
- **[LOW, needs_benchmark] Missing dense/unroll global flags — `:139-300`.** **Correction: these gate AFFINE global only** (consumers `:2049-2231` inside `global_affine_score_state_impl`), not linear `global_score_state` — narrower than stated. Validate on real AVX10 hardware.
- **[MEDIUM, needs_benchmark] scalar profile build — see §5.**

### arm-neon
- **[MEDIUM, high] No reduce_max — `arm_neon128.hpp:289-802` — confirmed.** Single-instruction `vmaxvq_s8/s16/s32` (already used at `:198/:283`); int64 `vgetq_lane_s64`×2 + `std::max`. Trivially safe, on the per-call critical path. (= §2)
- **[HIGH, high] width over-selection — preprocess; consumed `arm_neon128.hpp:1014` — confirmed.** = §1; i8=16 vs i16=8 lanes, the largest lever on NEON.
- **[LOW, needs_benchmark] No `any_nonzero`/`bit_or` → tier-2 lazy-F — `arm_neon128.hpp:289-802`; `farrar_fixed_kernel.hpp:3579-3633`.** Add `vorrq_*` + `vmaxvq_u32(...)!=0` to reach tier-1. **Correction: the per-segment `vorrq` is identical between tiers; the only delta is the per-*outer-iteration* test (`:3630`)** — `vgetq_lane_u64`×2+or vs one `umaxv`. Marginal, fires only in rare gap>0 shapes.
- **[LOW, needs_benchmark] i64 path — `arm_neon128.hpp:752-755 / 715-802`.** `max` emulation (`vbslq+vcgtq`) is correct/unavoidable. Only the i64 `reduce_max` (via `vgetq_lane_s64`) touches a generally-reached path; cold (UCS-2 routes to i32).

### arm-sve
- **[HIGH, high] width bound local-clamp — preprocess; `farrar_scalable_kernel.hpp:1247` clamp — confirmed.** = §1; i8=16 vs i16=8, gates whether int8 fast paths run at all.
- **[MEDIUM, high] `svqadd` wrongly gated behind SVE2; in-code comment is inverted — `arm_sve_backend.hpp:379-390/570-574 + comment 384-389` — confirmed, strongest SVE finding.** Verified in **both** GCC 16.1.0 (`aarch64-sve-builtins-base.def:170`, `none` predication = unpredicated vector-vector, base file) and clang-20 (`arm_sve.h:23777`, no SVE2 guard): unpredicated `svqadd_s8/16/32/64` is **base SVE**; SVE2 only adds the predicated `_m/_x/_z` forms. The `-msve` backend is needlessly denied it. Change guard to `__ARM_FEATURE_SVE`, add int8 overload, delete the backwards comment. Collapses 3-op `add_sentinel` (cmpeq+add+sel) to 1-op `svqadd` at NW-affine gap sites; also unblocks the SVE1 8-bit saturating-bias representation.
- **[HIGH, high] int8 (priority width) lacks `local_lazy_f_prefix_carry` / `shift_left_lanes_zero` — int8 struct `arm_sve_backend.hpp:192-288`; defined only int16 `:418-442`/int32 `:578-617` — confirmed.** SVE_BITS pinned to 128 (`CMakeLists.txt:278`) ⇒ lane_count is a constexpr 16, so a 4-stage `svext_s8(svdup(0),v,16-N)` carry with immediates `{15,14,12,8}` + `svmaxv_s8` is fully expressible. The densest, most-latency-sensitive width runs the up-to-16-step iterative loop while int16 (8 lanes) gets a 3-stage scan. (Line numbers in the finding are stale; constructs confirmed.)
- **[MEDIUM, high] int8/int64 never reach exact-fill dispatch — `farrar_scalable_kernel.hpp:1199-1212` — confirmed.** Auto-resolved for int8 once the carry above lands; int64 needs a 1-stage `svext_s64` carry. **Capability-completeness, not a measurable speedup** — the magic segment counts (128/256) need int8 queries ≥2048 chars, almost never produced.
- **[LOW, confirmed] int8 `add_sentinel` → `svqadd_s8` — `arm_sve_backend.hpp:279-287`.** 3→1 op at NW-affine gap sites (rhs≤0). Safe at `add_sentinel_negative_rhs` sites only (lhs bounded above INT8_MIN); do **not** substitute at generic `add_sentinel`.
- **[LOW, needs_benchmark] iterative lazy-F per-segment `svptest_any` — `farrar_scalable_kernel.hpp:1286-1305`.** Subsumed by the int8 prefix-carry. **Correction: the cited affine sites `:784-798` use `update_lazy_f` (unconditional, single stop-check), not per-segment any_greater** — claim holds only for the linear fallback at `:1293`.
- **[MEDIUM, needs_benchmark] scalar profile build (`:271-293`), uint32 offsets (`:350-354`) — see §5/§6.** Validate on graviton4 (no-emulation policy).

### riscv-rvv
**RVV is the worst-off backend: its SimdOps implement only `load/store/set1/zero/add/max/substitution`, so *every* striped primitive hits a heap-allocating `std::vector` fallback.** All RISC-V intrinsics below confirmed present in GCC 16.1.0 `rv64gcv`.
- **[HIGH, high] All striped primitives missing → heap fallbacks on the whole Farrar path — `linux_riscv64_rvv.hpp:25-215` — confirmed.** This is the dominant RVV inefficiency. Add natively:
  - `shift_left_zero` → `__riscv_vslide1up_vx(v,0,vl)` (1 instr; per-column `:1230` + per lazy-F iter `:1287`). **[HIGH, hot]**
  - `any_gt` → `__riscv_vfirst_m(__riscv_vmsgt_vv_i8m1_b8(a,b,vl),vl) >= 0` (per-segment in lazy-F `:1293`). **[HIGH, hot]** (`vfirst` cheaper than `vcpop`.)
  - `reduce_max` → `__riscv_vredmax_vs` + `__riscv_vmv_x_s` (`:1308`). **[MEDIUM]**
  - `shift_left_insert` → `vslide1up_vx(v,inserted,vl)` (NW `:941/:1400`). **[MEDIUM]**
  - `add_sentinel` → `vmseq_vx` + `vadd` + `vmerge_vvm` (NW affine `:879` etc). **[MEDIUM]**
  - `first_lane_vector` → `vmv_v_x(rest)` then `vmv_s_x(first)` (the finding's first `vslide1up` idea is wrong, as it admits). **[LOW]**
- **[HIGH, needs_benchmark] No `local_lazy_f_prefix_carry` → iterative lazy-F — `linux_riscv64_rvv.hpp`; gates `farrar_scalable_kernel.hpp:1199/1261`.** Implementable on runtime-VLEN RVV (`vslideup_vx` offset is a scalar, unlike `svext`'s immediate). **Three corrections demote it from "dominant":** (a) **SVE itself only implements this for int16/32/64, not int8 — the priority width — so both ISAs iterate there;** (b) the iterative loop early-exits for gap≤0 short strings (1-2 iters typical); (c) the "unlocks exact-fill" benefit doesn't apply to <100-char (segment_count never hits 128/256). **Semantic caveat:** `vslideup.vx` leaves the low `offset` destination lanes *undisturbed* (not zero-filled like `svext(svdup(0),…)`) — must seed/mask the vacated lanes to additive identity or the scan corrupts.
- **[HIGH, high] width over-selection — preprocess — confirmed.** = §1; 2× lane density, RVV's largest structural win, no ISA risk.
- **[MEDIUM, needs_benchmark] `add_sentinel` via `vsadd`, and 8-bit bias via `vsaddu` — `:54-56`.** **Re-scope: the `vsadd` win is NW/global affine** (those are the `add_sentinel` sites), not local SW; the bias variant is exploratory (§2).
- **[MEDIUM, needs_benchmark] LMUL=1 only — `:27/31` etc.** m2/m4/m8 could widen throughput, but **large LMUL overshoots <100-char queries (segment_count→1)** and the gain is microarch-dependent (128-bit datapaths just occupy the unit longer). A second m4 tier *selected by query length* is the only defensible form; measure on real RVV hardware.
- **[LOW, needs_benchmark] per-op `count` re-issues vsetvli (`:54-60`), `set1` in substitution (`:62-70`), non-pinned exact-fill gate.** All contingent on the native-ops fix; GCC's vsetvl-insertion likely already hoists the invariant vl. Inspect `-S` after the native ops land before acting.

### loongarch-lasx
- **[HIGH, high] `any_nonzero`/`trace_mask` 32-iter scalar movemask — `linux_loongarch64_lasx.hpp:39-49` — confirmed.** `detail::lane_mask` stores to stack + scalar loop; `any_nonzero` (`:375`), `any_gt` (`:355`) route through it, hit per lazy-F iteration (`farrar_fixed_kernel.hpp:3602`). Use `__lasx_xvmskltz_{b,h,w,d}` + `__lasx_xvpickve2gr_wu` (OR the two half-masks for `any_nonzero`; for `trace_mask` merge `low | high<<halfbits` to feed `countr_zero`). 1 vector + 2 GPR moves vs ~40 scalar ops.
- **[HIGH, high] No reduce_max — confirmed.** In-register `xvpermi_q(0x01)` + `xvmax` + `xvbsrl_v` tree already exists in this file's exact-fill kernels (`:187-194/287-293`); lift it + `xvpickve2gr_w/d`. (= §2)
- **[HIGH, high] uint32 (Chinese/UCS-2 first-class) lane pays both scalar reduce + scalar movemask — `linux_loongarch64_lasx.hpp:489-585` — confirmed.** Apply the two fixes above specifically to `SimdOps<uint32,int32>` (`xvmskltz_w`, the i32 reduce tree at `:287-293`). Largest realized benefit on the designated priority lane. (The `local_lazy_f_prefix_carry` third sub-fix only helps 1024-char UCS-2, off short-string.)
- **[HIGH, high] width over-selection — preprocess; payoff `:307/463` — confirmed.** = §1; 32→16 lanes.
- **[MEDIUM, high] No `local_lazy_f_prefix_carry` (all widths) → scalar carry — `farrar_fixed_kernel.hpp:1644-1663`; called `:3420` — confirmed.** Lift the in-file log-step (`:159-173/258-269`). **hot_path label is wrong** — only reachable via exact-fill (≥1024-char), not sub-100-char (which uses the §3 naive loop). Pair with §3 to extend the benefit to short strings.
- **[MEDIUM, needs_benchmark] lazy-F per-segment `bit_or`+`any_nonzero` — `:3588-3604`/`linux_loongarch64_lasx.hpp:371-377`.** Terminal-test fix (use `xvmskltz` not `lane_mask`) is safe; the per-segment restructuring may wash. Compounds with the `any_nonzero` fix.
- **[MEDIUM, needs_benchmark] int8 unsigned-saturating+bias — `:296-345` — see §2.**
- **[LOW, high] redundant `xvbitsel` in prefix carry — `:162-173/261-269` — confirmed.** `shift_left_insert<N>(prefix,zero)` → `shift_left_zero<N>(prefix)`; drops one `xvbitsel_v` + one `xvld` constant per stage. 1024-char exact-fill path only.
- **[LOW, high] int64 path — `:587-676` — confirmed native (`xvmax_d/xvadd_d`), only reduce/movemask spill.** Low-priority cleanup (UCS-2 → i32).

### loongarch-lsx
- **[HIGH, high] `any_nonzero`/`any_gt`/`trace_mask` stack-spill — `linux_loongarch64_lsx.hpp:38-49/166-188` — confirmed.** Use `__lsx_bnz_v(v)!=0` and `any_gt = __lsx_bnz_v(greater_mask(a,b))!=0` (`VSETNEZ.V` → CR branch, no GPR roundtrip); `trace_mask` via `__lsx_vmskltz_b/h/w/d`. The genuine hot cost is the single `any_nonzero` at `:3602` (the cited `:3424`/bounded-scan sites are in exact-fill, not the sub-100-char path).
- **[HIGH, high] No reduce_max — confirmed.** `vmax_{b,h,w,d}` (signed) over `vbsrl_v` shifts 8/4/2/1 + `vpickve2gr_*`. (= §2)
- **[HIGH, high] int8 unsigned-saturating+bias — `:82-198` — needs_benchmark.** See §2; `vsadd_bu`/`vmax_bu`/`vseq_b`/`bnz_*` exist. Architectural, gated on §1.
- **[MEDIUM, needs_benchmark] vectorize profile build — `:190-197` primitive — see §5.**
- **[LOW, high] redundant `vbitsel_v` in prefix carry — `:72-78/141-388` — confirmed.** `shift_left_insert<N>(.,zero)` → `shift_left_zero<N>`. Off priority (affine/exact-fill).
- **[LOW, uncertain] no `*_raw` exact-fill kernels — `:88/206/320`.** Advertises segment64/128/256 but uses generic; only aligned-length queries hit it. Primary action = inherit the reduce_max fix; LASX-port deprioritized.

### powerpc-vsx
*(VSX already defines reduce_max and local/global prefix carries — it's the most-complete non-AVX512 backend; only minor items.)*
- **[MEDIUM, high] `any_nonzero`/`any_mask` 2× `vec_extract` (VSR→GPR) — `linux_powerpc64_vsx.hpp:126-129` — confirmed.** Replace with `vec_any_ne(reinterpret<u64>(v), vec_splats(0ULL))` (single `vcmpequd.`+CR read); the backend already uses `vec_any_gt` this way (`:269`). Removes 2 `mfvsrd` (~5cyc each) per lazy-F convergence check (`:3602`). Apply to both for all widths.
- **[LOW, needs_benchmark] dense lazy-F builds greater_mask vector then any_nonzero — `:265-293`/`:3579-3605`.** Per-segment dotted `vec_any_gt` may serialize on CR6; **shared-kernel change, not VSX-local**. Fix the trailing `any_nonzero` (above) first.
- **[LOW, uncertain] reduce_max redundant `zero()` splats / shift-vs-rotate — `:305-333` etc.** The "3 redundant `xxlxor`" are almost certainly already CSE'd by GCC. Real value: `vec_sld(v,v,N)` rotate drops the zero operand **and** makes reduce_max correct for all-negative inputs (latent NW edge) — treat as correctness-hardening, not perf.
- **[LOW, needs_benchmark] global `add_sentinel` extra cmp+sel — `:179-263`.** `vec_adds` (also valid for **int32** `vaddsws`, not just 8/16 — the header comment understates it; int64 has none). Changes out-of-range wrap→clamp; validate exact NW scores on real POWER8.

### generic-scalar
- **[MEDIUM, high] 2 heap row buffers per call; batch mallocs 2N — `generic.hpp:72-73/141-142` — confirmed.** `thread_local` buffer pair, resize-grow + `std::fill`; hoist through `matrix_scores_dispatch`. = §4, worst-hit here (no profile/arena to amortize).
- **[LOW, needs_benchmark] matrix-mode per-cell gather — `scorer.hpp:62-68`.** The `q*stride` multiply is **loop-invariant w.r.t. the column loop and Token(uint8)≠Cell, so LICM almost certainly already hoists `qrow = matrix + q*stride`** under -O3 (TBAA permits it). Expected outcome: no win. The residual dependent gather is inherent.

### Backends with nothing notable
None — every backend had at least the cross-cutting reduce_max / width-bound items. (LSX/LASX/RVV/SVE/SSE/AVX10 had the most actionable backend-specific work; VSX and AVX512bwvl are the most complete.)

---

## Checked and dismissed

- **AVX2 `shift_left_zero_256` (4-op cross-lane shift)** — irreducible on AVX2 (`permute2x128` mandatory; `alignr_epi8` is per-128-lane). No fix.
- **AVX2 int64 `max` (`cmpgt_epi64+blendv`), AVX2 aligned load/store** — already optimal; AVX2 *has* `_mm256_cmpgt_epi64` (SSE4.1 does not).
- **SSE4.1 int64 substitution/add_sentinel (`cmpeq_epi64+blendv`)** — optimal; guard against regression.
- **NEON / LSX / LASX int64 `max`** — correctly emulated (`vbslq+vcgtq` / native `xvmax_d`,`vmax_d`); no native horizontal `vmaxvq_s64` exists.
- **PPC reduce_max "redundant zero splats"** — likely already CSE'd; only the rotate form has (correctness) value.
- **VSX has reduce_max + both prefix carries** — corrects the shared-finding claim that it lacked them (the dead `powerpc_vsx_kernel.hpp` was misattributed).
- **`affine_scalable_kernel.hpp` / `affine_fixed_kernel.hpp` anti-diagonal `run_kernel`** — **DEAD for all backends** (including SWAR: `dispatch_affine_score`/`_traceback` at `swar.hpp:437/507` have no callers; live affine routes through striped `farrar_*_kernel::affine_score`). The 11× `={}` 2048-bit scratch zeroing (`affine_scalable_kernel.hpp:339-351`) is never paid. **Recommend deleting** to cut compile time/instantiations — not a runtime fix.
- **Generic-scalar matrix per-cell multiply** — LICM almost certainly already hoists it; no win expected.
- **RVV `set1`-in-substitution, vsetvli-per-op** — compiler likely already handles (loop-invariant / vsetvl-insertion); verify `-S` only after native ops land.
- **`apply_forced_kernel_bits` re-derives width (`preprocess.hpp:237-248`)** — real dead compute on pinned wide paths; **also a latent mis-validation** (validates forced width against *automatic* not *pinned* bits, silently ignoring e.g. width=16 on uint32 tokens). Low severity; fix tightens validation.
- **`byte_symbol_count` computed but unused by score width (`farrar_preprocess.hpp:107-147`)** — confirmed dead for score-only width selection; make lazy (it's a public field — don't delete).

---

## Top changes to implement first

Hot-path / short-string items weighted highest; sure wins before benchmark-gated ones.

1. **Local-SW-aware width bound** (§1 / prior #1) — **CONFIRMED, biggest lever.** Thread `is_local` into `compute_score_bound`/`select_score_bits`; use `min(q,t)*max(magnitude(match), mismatch>0?magnitude(mismatch):0)`, NW keeps the global bound, fall back when any gap>0. **Payoff: ~2× lane density (int8 vs int16) on the priority workload across all 9 backends; precondition for every int8 fast path.** Effort: low-medium (preprocess only). Do this first.

2. **`reduce_max` members** (§2 / prior fallback) — **CONFIRMED.** Add per-ISA in-register reductions to SSE/AVX2/NEON/LASX/LSX/AVX10-256/AVX10-512 (single-instr on NEON `vmaxvq`; lift existing trees on LASX/LSX; `extracti128`-fold on AVX2; `minpos`-complement on SSE int16). Effort: low per backend, mechanical. **Payoff: removes a store-forwarding stall + 8-32 scalar compares from the tail of every score-only call.** Skip VSX/AVX512bwvl (already have it).

3. **RVV native striped primitives** (RVV §R1) — **CONFIRMED, multiplicative.** `vslide1up`/`vfirst+vmsgt`/`vredmax+vmv.x.s`/`vmerge`/`vslide1up(insert)`/`vmv_s_x`. Effort: low-medium (one file, intrinsics confirmed). **Payoff: removes O(target×lane_count) malloc/free + register spills from the entire RVV Farrar hot path — by far the largest RVV win.**

4. **Generic `score_state` log-step lazy-F** (§3) — **CONFIRMED.** Swap the naive O(lane_count) loop for `local_lazy_f_prefix_carry` + bounded scan (already used by exact-fill). Effort: medium (shared kernel, one site, broad reach). **Payoff: lazy-F correction from O(lane_count·segment_count) worst-case to O(log lane_count)+1 bounded pass on every short-string call; biggest on gap-run inputs** — but the existing `gap≤0` early-exit caps typical iterations, so *measure the win on adversarial inputs* even though the change is never slower.

5. **Per-call state arena / `thread_local`** (§4 / prior #5) — **CONFIRMED.** Mirror the matrix path's `thread_local PreparedScoreState` + in-place prepare for the linear score path; hoist out of batch loops. Effort: medium. **Payoff: eliminates ~4-5 allocs + zero-fills per short-string call; large relative gain on tiny-DP cdist/`*_scores` sweeps.** Generic-scalar (§4, `generic.hpp:72-73`) is a quick subset — do it too.

6. **SVE `svqadd` guard fix** (SVE) — **CONFIRMED, two-toolchain-verified.** Change `__ARM_FEATURE_SVE2`→`__ARM_FEATURE_SVE`, add int8 overload, delete the inverted comment. Effort: trivial. **Payoff: 3-op→1-op `add_sentinel` on the SVE1 NW-affine inner loop; unblocks the SVE1 saturating-bias path.** (NW, not priority — but near-zero cost and a clear correctness/clarity fix.)

7. **LASX/LSX scalar-movemask → `xvmskltz`/`bnz_v`** (LASX/LSX) — **CONFIRMED.** Replace `detail::lane_mask` spill in `any_nonzero`/`any_gt`/`trace_mask`. Effort: low. **Payoff: removes a ~16-32-lane store+scalar loop from every lazy-F convergence check; on LASX this directly serves the Chinese/UCS-2 uint32 priority lane.**

8. **SSE prefix-carry redundant `blendv` + LASX/LSX prefix-carry `xvbitsel`/`vbitsel`** — **CONFIRMED, trivial.** `shift_left_insert<N>(.,zero)` → `shift_left_zero<N>`. Effort: trivial. **Payoff: drops one port-5 blend + one constant load per log-step; small but free.** (SSE is hot exact-fill; LASX/LSX are 1024-char only.)

**Honest needs_benchmark flags (do NOT ship as sure wins):** the 8-bit unsigned-saturating+bias representation (§2/§7, every backend) — large potential but architectural, gated on #1, and adds inner-loop ops; vectorized profile build (§5) — benefit scales with *long* queries, not the priority case; uint32 offsets (§6) — L1-resident no-op for short strings; RVV LMUL>1 — microarch-dependent and overshoots short queries; AVX2/AVX512 NW saturating-`add_sentinel` and masked-store flags — NW-only, marginal. **Dead-code cleanup (compile-time only):** delete the anti-diagonal `run_kernel` engines in `affine_{fixed,scalable}_kernel.hpp`.