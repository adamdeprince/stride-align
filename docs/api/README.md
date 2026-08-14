# API

stride-align exposes its native algorithms through Python, R, DuckDB,
PostgreSQL, and Memgraph. Every algorithm below includes one call in each
interface. Use the selector on any example; all selectors on this page and the
homepage stay synchronized.

The examples assume `import stride_align as sa` in Python,
`library(stridealign)` in R, or the Python database clients shown in each
database panel. See the [homepage installation walkthrough](../../index.html#install),
[download the DuckDB extension](https://distribution.goblinreactor.com/stride-align/duckdb/),
or [download the Memgraph query module](https://distribution.goblinreactor.com/stride-align/memgraph/)
matching the server platform and CPU. PostgreSQL 16–18 installation uses the
[PGXS build guide](../../bindings/postgres/).

The API is organized by algorithm, not by language adapter. Python and R use
ordinary function calls. DuckDB and PostgreSQL expose `stride_*` SQL functions;
Memgraph exposes `stride_align.*` Cypher functions and a streaming `cdist`
procedure. Database examples use Python only to connect, load data, and submit
the native SQL or Cypher operation. Compatibility facades are intentionally
outside this native cross-language reference.

## Vectorization semantics

Let `Q` be the number of queries, `T` the number of targets, and `K` the
requested result limit. The function name determines both how much of the
`Q × T` comparison grid is retained and the shape returned to the caller.

| Operation | Logical input | Logical output |
| --- | --- | --- |
| `*_scores` / `*_similarities` | one query and `T` targets | `T` scores in target order |
| `*_best` | one query and `T` targets | one match, or no match for an empty candidate set |
| `*_top_k` / `extract` | one query and `T` targets | at most `K` matches across those targets |
| `cdist` | `Q` queries and `T` targets | dense `Q × T` score matrix, or `Q·T` streamed pair rows in Memgraph |
| `cdist_above_threshold` | conceptual `Q × T` grid | only pair records meeting the threshold |
| `cdist_top_k` | conceptual `Q × T` grid | at most `K` pair records **across the whole grid** |
| `cdist_top_k_per_query` | conceptual `Q × T` grid | `Q` groups, each containing at most `K` targets |

The host languages preserve those logical shapes using their native data
types:

| Host | Collection and index conventions |
| --- | --- |
| Python | Score vectors and dense `cdist` results are NumPy arrays. Rankings are tuples/lists; threshold and per-query forms are iterators. Match indices are zero-based. |
| R | Results use numeric vectors, matrices, data frames, and lists. Match indices are one-based. Pair scorers also accept equal-length character vectors or broadcast a length-one side. |
| DuckDB | Results use `LIST`, nested `LIST`, and `STRUCT` values. Match index fields are zero-based, while SQL list subscripts are one-based. Pair functions applied to table columns run through DuckDB's vectorized chunk machinery; list-valued functions can additionally batch a candidate collection within each row. |
| PostgreSQL | Results use native arrays and `jsonb` match records. SQL array subscripts are one-based; match records retain zero-based source indices. Its native batch functions accept `text[]` collections directly. |
| Memgraph | Scalar functions compose over Cypher rows. There are no plural, extraction, or top-k function names. `cdist` is a batch read procedure yielding zero-based `query_index`, `target_index`, and `score` fields in row-major order, so consumers need not materialize a matrix. |

For non-matrix string scorers, threshold and all-pairs top-k selection use a
normalized similarity scorer; dense `cdist` also supports raw distances. The
scorer-specific match, mismatch, gap, prefix, cutoff, and substitution-matrix
options carry through the collection forms that support that scorer.

> `cdist_top_k(..., k=3)` returns no more than three matches total, even if
> there are thousands of queries. `cdist_top_k_per_query(..., k=3)` gives each
> query its own competition and can therefore return as many as `3 × Q`
> matches. Choose the first for global record linkage and the second for
> nearest neighbors for every query.

In Memgraph the same selections are expressed after `CALL stride_align.cdist`:
`WHERE` applies a threshold, `ORDER BY ... LIMIT` selects a global top-k, and a
grouped `collect(...)[..k]` gives each query its own top-k. That keeps selection
inside Cypher while the procedure continues to produce pair rows incrementally.

<!-- stride-vectorization-guide -->

## Algorithms

<!-- stride-api-catalog -->

Detailed parameter, return-value, matrix, and traceback semantics are linked
from each algorithm above.
