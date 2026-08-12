# stride-align for DuckDB

Fast string matching and sequence alignment, directly in DuckDB SQL. Load the
extension and call `stride_*` functions anywhere you would use another scalar
function.

- **Easy to use:** pass ordinary `VARCHAR` columns and get numeric scores.
- **Unicode included:** ASCII, accents, CJK, combining marks, and emoji use the
  same SQL functions.
- **Fast on real queries:** functions run over DuckDB vectors through a native
  SIMD package selected for the target machine.

The extension has no Python or nanobind runtime dependency. Python appears in
one example below only as a convenient way to orchestrate DuckDB.

## Quick start

Choose the extension artifact matching your DuckDB version, platform, and CPU
package. Unsigned local extensions require DuckDB's sideload opt-in.

[Browse published DuckDB packages](https://distribution.goblinreactor.com/stride-align/duckdb/).

From the DuckDB CLI:

```sh
duckdb -unsigned
```

```sql
LOAD '/absolute/path/to/stride_align.duckdb_extension';

SELECT stride_levenshtein('kitten', 'sitting');
-- 3

SELECT stride_levenshtein_similarity('你好世界', '你好世间');
-- 0.75
```

Or from Python:

```python
import duckdb

db = duckdb.connect(config={"allow_unsigned_extensions": "true"})
extension = "/absolute/path/to/stride_align.duckdb_extension"
db.execute(f"LOAD '{extension}'")

score = db.execute(
    "SELECT stride_levenshtein_similarity(?, ?)",
    ["你好世界", "你好世间"],
).fetchone()[0]
```

## Use it in a query

The functions compose with normal DuckDB SQL:

```sql
WITH records(name) AS (
  VALUES ('Martha'), ('Marhta'), ('Arthur')
)
SELECT
  name,
  stride_jaro_winkler(name, 'Martha') AS similarity
FROM records
ORDER BY similarity DESC;
```

Use the same pattern with table columns for fuzzy search, deduplication,
record linkage, ranking, or biological sequence scoring.

## SQL functions

All two-string functions accept `VARCHAR` and propagate SQL `NULL`. Hamming
functions require equal-length strings.

| Function | Result |
| --- | --- |
| `stride_levenshtein(a, b)` | Levenshtein distance |
| `stride_levenshtein_similarity(a, b)` | normalized similarity |
| `stride_osa(a, b)` / `stride_osa_similarity(a, b)` | restricted Damerau-Levenshtein |
| `stride_true_damerau_levenshtein(a, b)` | unrestricted Damerau-Levenshtein |
| `stride_indel(a, b)` / `stride_indel_similarity(a, b)` | insert/delete distance |
| `stride_hamming(a, b)` / `stride_hamming_similarity(a, b)` | equal-length Hamming |
| `stride_jaro(a, b)` / `stride_jaro_winkler(a, b)` | Jaro similarities |
| `stride_smith_waterman(a, b)` | local-alignment score, defaults `2/-1/-1` |
| `stride_needleman_wunsch(a, b)` | global-alignment score, defaults `2/-1/-1` |
| `stride_*_affine(a, b)` | affine-gap SW/NW score, defaults `2/-1/-2/-1` |
| `stride_align_simd_level()` | selected package profile |

DuckDB exposes the native stride-align SQL surface. The RapidFuzz, Parasail,
Jellyfish, and TheFuzz compatibility APIs are Python-only migration aids.

## Packages

DuckDB extensions are tied to their DuckDB version and platform. stride-align
also publishes separate CPU packages so deployment and benchmarking stay
explicit:

| Profile | Intended package |
| --- | --- |
| `generic` | portable platform baseline |
| `avx2` | x86-64-v3 / AVX2 |
| `avx512bwvl` | x86-64-v4 / AVX-512 F/BW/VL/DQ |
| `neon` | AArch64 NEON |
| `la464_lsx` | Loongson LA464, LSX only |
| `la464_lasx` | Loongson LA464, LASX |
| `la664_lasx` | Loongson LA664, LASX |
| `native` | local benchmark/development build only |

Check the loaded package at any time:

```sql
SELECT stride_align_simd_level();
```

Loading an ISA-specific package on an incompatible CPU is not supported.

## Build from source

Pin a DuckDB source checkout to the exact version you plan to load the
extension into. From the stride-align repository root:

```sh
cmake -S /path/to/duckdb -B build/duckdb-native -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_EXTENSIONS_ONLY=OFF \
  -DDUCKDB_EXTENSION_CONFIGS="$PWD/bindings/duckdb/extension_config.cmake" \
  -DSTRIDE_ALIGN_DUCKDB_SIMD=native
cmake --build build/duckdb-native \
  --target stride_align_loadable_extension duckdb
```

The artifact is normally written to
`build/duckdb-native/extension/stride_align/stride_align.duckdb_extension`.
The extension config uses `DONT_LINK`, so the resulting DuckDB executable is
also suitable for testing the artifact through the real sideload path.

DuckDB 1.5.5 does not infer LoongArch64 correctly when it generates extension
metadata. LoongArch64 package builds must also pass:

```sh
-DDUCKDB_EXPLICIT_PLATFORM=linux_loongarch64
```

Validate the resulting DuckDB CLI with `PRAGMA platform`; it must return
`linux_loongarch64` before the extension is packaged.

## Python parity tests

The Python test harness sideloads the real extension into the pinned DuckDB
client and compares every SQL scorer with its native `stride_align` Python
counterpart. It also exercises vector-sized inputs, Unicode, `NULL`
propagation, and error translation.

Place a package artifact under the normal distribution layout and run:

```sh
pytest tests/duckdb
```

For an artifact elsewhere, or when several SIMD packages are present, select
it explicitly:

```sh
STRIDE_ALIGN_DUCKDB_EXTENSION=/absolute/path/to/stride_align.duckdb_extension \
  pytest tests/duckdb

STRIDE_ALIGN_DUCKDB_SIMD=avx2 pytest tests/duckdb
```

An explicitly selected missing or unloadable artifact fails the suite. If no
artifact has been built at all, these tests report a skip so the Python-only
suite remains usable from a clean source checkout.
