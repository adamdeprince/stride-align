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

**[Download stride-align packages for DuckDB](https://distribution.goblinreactor.com/stride-align/duckdb/)**

## Download and install

Choose the extension artifact matching your DuckDB version, platform, and CPU
package. Unsigned local extensions require DuckDB's sideload opt-in.

From the DuckDB CLI:

```sh
duckdb -unsigned
```

```sql
LOAD '/absolute/path/to/stride_align.0.6.0.duckdb_extension';

SELECT stride_levenshtein('kitten', 'sitting');
-- 3

SELECT stride_levenshtein_similarity('你好世界', '你好世间');
-- 0.75
```

Or from Python:

```python
import duckdb

db = duckdb.connect(config={"allow_unsigned_extensions": "true"})
extension = "/absolute/path/to/stride_align.0.6.0.duckdb_extension"
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

## Vectorized nearest lookup

The [Bible lookup and spell-check demo](../../demo/duckdb_vectorized_lookup.py)
keeps candidates in a DuckDB table, collects them into an ordered list, and
hands that complete vector to stride-align's optimized top-one operation:

```sql
WITH corpus AS (
  SELECT list(lower(verse) ORDER BY reference, verse) AS verses
  FROM demo_bible
)
SELECT stride_extract_best(lower(?), verses, 'needleman_wunsch_normalized')
FROM corpus;
```

The result contains the winning target, score, and zero-based list index. The
full demo uses that index to recover the corresponding reference and original
verse. The spell-check mode tokenizes the input in SQL, collects the dictionary
column into a list, and calls the same batch top-one operation for each token.
Python only sideloads the extension, submits each complete query, and prints
its result.

```sh
export STRIDE_ALIGN_DUCKDB_EXTENSION=/absolute/path/to/stride_align.0.6.0.duckdb_extension
python3 demo/duckdb_vectorized_lookup.py bible
python3 demo/duckdb_vectorized_lookup.py spell
```

## SQL functions

All string functions accept ordinary UTF-8 `VARCHAR` values. List-valued APIs
use `VARCHAR[]`, and every function can be applied to DuckDB columns as well as
literals. Hamming functions require equal-length strings.

| Family | Examples |
| --- | --- |
| Pairwise edit and alignment | `stride_levenshtein_score`, `stride_indel_score`, `stride_jaro_winkler_similarity`, `stride_smith_waterman_score` |
| One-to-many vectors | `stride_levenshtein_scores(query, targets)`, `stride_jaro_similarities(query, targets)`, `stride_wratios(query, targets)` |
| Ranking | `stride_levenshtein_top_k`, `stride_jaro_best`, `stride_extract` |
| All-pairs | `stride_cdist`, `stride_cdist_above_threshold`, `stride_cdist_top_k`, `stride_cdist_top_k_per_query` |
| Text algorithms | LCS, n-gram, Ratcliff-Obershelp, token ratios, WRatio, and Monge-Elkan |
| Phonetics | Soundex, Metaphone, NYSIIS, Match Rating, Caverphone, Cologne, Daitch-Mokotoff, Double Metaphone, and Beider-Morse |
| Sequences | DTW, local/global paths and CIGARs, BLOSUM/PAM/nucleotide matrices, and custom substitution matrices |

The complete catalog lists all 167 registered SQL function names and the
families that supply their 413 overloads: [DuckDB SQL function catalog](FUNCTIONS.md).
Inside DuckDB, inspect the exact signatures in the package you loaded with:

```sql
SELECT function_name, parameters, parameter_types, return_type
FROM duckdb_functions()
WHERE function_name LIKE 'stride_%'
ORDER BY function_name, parameter_types;
```

The ranking and cdist functions are native bounded operations. They apply
length bounds, adaptive score cutoffs, and bounded heaps while scanning; a
query does not need to materialize and sort a full cross-product first.

```sql
SELECT stride_levenshtein_top_k(
  'kitten', ['sitting', 'kitten', 'bitten', 'kitchen'], 2
);

SELECT stride_cdist_top_k(
  ['kitten', 'sitting'],
  ['kitten', 'bitten', 'kitchen'],
  'jaro_winkler',
  4
);
```

Built-in matrices are addressed by name. Custom matrices are SQL values and
can be passed to scalar, vector, path, cdist, threshold, and top-k functions.

```sql
WITH matrix AS (
  SELECT stride_identity_matrix('ACGT', 5, -4, 'N', 'DNA', -5) AS value
)
SELECT stride_smith_waterman_matrix_score('ACGT', 'AGGT', value)
FROM matrix;
```

Use `stride_matrix_available()` to list the embedded BLOSUM, PAM, nucleotide,
text, and keyboard matrices. `stride_substitution_matrix`,
`stride_matrix_from_ncbi_text`, `stride_keyboard_from_confusion_counts`, and
`stride_keyboard_from_npy` build application-specific matrices.

DuckDB exposes the native stride-align surface. RapidFuzz, Parasail, Jellyfish,
and TheFuzz compatibility facades are intentionally not registered.

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

Published artifacts use the filename
`stride_align.<project-version>.duckdb_extension`; release 0.6.0 is therefore
`stride_align.0.6.0.duckdb_extension`. The version must match the Python and R
release in `pyproject.toml`. Keep `stride_align` before the first dot because
DuckDB derives the extension entrypoint name from that segment.

`repository-index.html`, `repository-manifest.json`, and
`repository-SHA256SUMS` are the checked-in publication metadata for the current
release.

## Build from source

Pin a DuckDB source checkout to the exact version you plan to load the
extension into. From the stride-align repository root:

```sh
cmake -S /path/to/duckdb -B build/duckdb-native -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_EXTENSIONS_ONLY=OFF \
  -DEXTENSION_STATIC_BUILD=ON \
  -DDUCKDB_EXTENSION_CONFIGS="$PWD/bindings/duckdb/extension_config.cmake" \
  -DSTRIDE_ALIGN_DUCKDB_SIMD=native
cmake --build build/duckdb-native \
  --target stride_align_loadable_extension duckdb
```

The artifact is normally written to
`build/duckdb-native/extension/stride_align/stride_align.duckdb_extension`.
Published packages rename that raw build output to
`stride_align.0.6.0.duckdb_extension`, using the same release version as the
Python and R packages.
The extension config uses `DONT_LINK`, so the resulting DuckDB executable is
also suitable for testing the artifact through the real sideload path.
Pass `EXTENSION_STATIC_BUILD=ON` explicitly on fresh GNU/Linux builds. DuckDB
uses it to compile its static library into independently discardable function
and data sections, keeping the sideloaded package from retaining unused DuckDB
core code.

DuckDB 1.5.5 does not infer LoongArch64 correctly when it generates extension
metadata. LoongArch64 package builds must also pass:

```sh
-DDUCKDB_EXPLICIT_PLATFORM=linux_loongarch64
```

Validate the resulting DuckDB CLI with `PRAGMA platform`; it must return
`linux_loongarch64` before the extension is packaged.

## Python parity tests

The Python test harness sideloads the real extension into the pinned DuckDB
client and compares the complete native SQL surface with `stride_align`'s
Python results. It covers scalar and vector calls, optimized cdist/ranking,
Unicode, `NULL`, paths, DTW, phonetics, and built-in/custom matrices.

`python-test-parity.tsv` maps every non-compatibility Python test module to its
DuckDB adaptation. The checker fails when a new native Python suite is added
without a DuckDB counterpart:

```sh
bindings/duckdb/check-python-test-parity.py
```

`native-api.tsv` separately maps every public Python/R capability to its SQL
form and verifies that each named function is present in the loaded extension:

```sh
STRIDE_ALIGN_DUCKDB_EXTENSION=/absolute/path/to/stride_align.0.6.0.duckdb_extension \
  bindings/duckdb/check-native-api.py
```

Place a package artifact under the normal distribution layout and run:

```sh
pytest tests/duckdb
```

For an artifact elsewhere, or when several SIMD packages are present, select
it explicitly:

```sh
STRIDE_ALIGN_DUCKDB_EXTENSION=/absolute/path/to/stride_align.0.6.0.duckdb_extension \
  pytest tests/duckdb

STRIDE_ALIGN_DUCKDB_SIMD=avx2 pytest tests/duckdb
```

An explicitly selected missing or unloadable artifact fails the suite. If no
artifact has been built at all, these tests report a skip so the Python-only
suite remains usable from a clean source checkout.
