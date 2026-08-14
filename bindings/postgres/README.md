# stride-align for PostgreSQL

The PostgreSQL extension exposes stride-align's complete native API on
PostgreSQL 16, 17, and 18. It registers the same 167 function names as the
DuckDB package, including scalar scores, one-to-many scoring, `cdist`, global
and per-query top-k operations, paths and CIGARs, phonetics, DTW, and
substitution-matrix operations. RapidFuzz, TheFuzz, Parasail, and Jellyfish
compatibility shims are intentionally not included.

PostgreSQL extensions are tied to a server major version. Build and package a
separate binary for each of 16, 17, and 18; the implementation and generated
SQL catalog are shared.

## Build and install

Select the target server with its `pg_config`:

```sh
make clean
make \
  PG_CONFIG=/usr/lib/postgresql/18/bin/pg_config \
  STRIDE_ALIGN_POSTGRES_SIMD=avx2 \
  -j8
sudo make \
  PG_CONFIG=/usr/lib/postgresql/18/bin/pg_config \
  STRIDE_ALIGN_POSTGRES_SIMD=avx2 \
  install
```

Then enable it in a database:

```sql
CREATE EXTENSION stride_align;
```

Available package profiles are `generic`, `native`, `avx2`, `avx512bwvl`,
`neon`, `power8_vsx`, `la464_lsx`, `la464_lasx`, and `la664_lasx`. The named
profiles set both the stride-align specialization defines and the matching
compiler CPU flags, so the compiler's auto-vectorizer targets the same machine
as the explicitly selected templates.

## Use

Pairwise and vector operations are ordinary SQL functions:

```sql
SELECT stride_levenshtein('kitten', 'sitting');

SELECT stride_levenshtein_scores(
  'kitten',
  ARRAY['kitten', 'sitting', 'bitten']
);

SELECT stride_cdist_top_k(
  ARRAY['kitten', 'sitting'],
  ARRAY['kitten', 'bitten', 'smitten'],
  'jaro',
  3
);
```

Matrix names work directly. Custom matrices are opaque `bytea` values that
can be stored, passed between functions, and inspected with
`stride_matrix_info`:

```sql
SELECT stride_smith_waterman_matrix_score(
  'HEAGAWGHEE', 'PAWHEAE', 'blosum62', -4
);

WITH matrix_value AS (
  SELECT stride_substitution_matrix(
    'DNA_CUSTOM',
    'ACN',
    ARRAY[[5,-4,-4],[-4,5,-4],[-4,-4,-4]]::integer[],
    -5,
    'N'
  ) AS value
)
SELECT stride_matrix_info(value) FROM matrix_value;
```

Query PostgreSQL for the authoritative catalog installed on a server:

```sql
SELECT DISTINCT p.proname
FROM pg_proc AS p
JOIN pg_namespace AS n ON n.oid = p.pronamespace
WHERE n.nspname = 'public' AND p.proname LIKE 'stride_%'
ORDER BY p.proname;
```

## Native server encodings

The extension does not convert PostgreSQL text to UTF-8. It asks PostgreSQL
for the database encoding and its character boundaries, then preserves the
original character byte sequences as equality tokens:

- SQL_ASCII and every single-byte server encoding are borrowed directly into
  an 8-bit channel, including high-bit characters.
- ASCII text in a multibyte database takes the SIMD ASCII fast path and is
  likewise borrowed as 8-bit data.
- Short multibyte pairs use direct 16- or 32-bit native tokens.
- Long multibyte pairs use joint Swiss-table cardinality packing into the
  narrowest 8-, 16-, or 32-bit channel.
- Streaming algorithms, where a later string cannot be inspected while the
  earlier string is prepared, use 32-bit native tokens.

The integration suite creates real PostgreSQL servers, loads the extension on
all three supported majors, and performs a PostgreSQL 18 sweep across every
valid server encoding:

```sh
pytest tests/postgres
```

Set `STRIDE_ALIGN_PG_CONFIGS` to an operating-system-path-separated list when
the desired `pg_config` executables are outside the standard locations.
