# stride-align for Memgraph

The Memgraph query module exposes stride-align 0.6.0 as ordinary Cypher
functions and a streaming `cdist` procedure. Pairwise calls compose naturally
with `MATCH`, `UNWIND`, projections, filters, and aggregations:

```cypher
MATCH (candidate:Verse)
RETURN candidate.text,
       stride_align.jaro_winkler($query, candidate.text) AS similarity
ORDER BY similarity DESC
LIMIT 10;
```

The module deliberately has no one-to-many list functions, extraction helpers,
or top-k functions. Cypher already supplies the row machinery for those cases.
RapidFuzz, TheFuzz, Parasail, and Jellyfish compatibility shims are also absent.
The native scalar catalog is listed in [FUNCTIONS.md](FUNCTIONS.md).

Prebuilt stride-align 0.6.0 modules for Memgraph 3.10.1 are available from the
[self-hosted package index](https://distribution.goblinreactor.com/stride-align/memgraph/).
Linux x86-64 generic and AVX2 packages target the `manylinux_2_28` ABI (glibc
2.28 or newer) and are runtime tested. The macOS arm64/NEON package targets
macOS 13 or later and is build tested against the same public Memgraph API
header.

## Build and install

The current package target is Memgraph 3.10.1. Build against the public
`mg_procedure.h` shipped with the matching Memgraph installation or container:

```sh
cmake -S bindings/memgraph -B build/memgraph \
  -DSTRIDE_ALIGN_MEMGRAPH_API_DIR=/usr/include/memgraph \
  -DSTRIDE_ALIGN_MEMGRAPH_SIMD=avx2 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/memgraph -j8
```

Copy `build/memgraph/stride_align.so` into Memgraph's query-module directory,
normally `/usr/lib/memgraph/query_modules`, and start or reload Memgraph. The
file name supplies the `stride_align` namespace.

Downloaded artifacts retain the stride-align version in their filename. Rename
the selected package while installing it:

```sh
sudo install -m 0755 stride_align-0.6.0.so \
  /usr/lib/memgraph/query_modules/stride_align.so
```

A Memgraph container on macOS is still a Linux server process. Match a package
to the container's operating system and architecture rather than the macOS
host; the native macOS artifact is only for a native macOS Memgraph process.

For a container, mount the module read-only:

```sh
podman run --rm -p 7687:7687 \
  -v "$PWD/build/memgraph/stride_align.so:/usr/lib/memgraph/query_modules/stride_align.so:ro" \
  docker.io/memgraph/memgraph:3.10.1
```

Available build profiles are `generic`, `native`, `avx2`, `avx512bwvl`,
`neon`, `power8_vsx`, `la464_lsx`, `la464_lasx`, and `la664_lasx`. Each named
profile selects both the matching stride-align templates and compiler CPU
flags, so package builders should use the profile of the deployment machine.

## Scalar functions

Functions return one value for one input row:

```cypher
RETURN stride_align.levenshtein('kitten', 'sitting');

RETURN stride_align.smith_waterman_score(
  'ACCGT', 'ACG', 2, -1, -2, -1
);

WITH stride_align.smith_waterman_path('ACCGT', 'ACG') AS path
RETURN path.score, path.cigar, path.aligned_query, path.aligned_target;
```

UTF-8 input is handled directly. ASCII takes the narrow fast path, while other
Unicode input is represented in the narrowest safe score channel for each
pair. The public Memgraph C API exposes strings as NUL-terminated data, so—as
with other native MGP modules—embedded NUL characters cannot be distinguished
from the end of a string.

## Streaming `cdist`

`cdist` is a procedure rather than a matrix-valued function. It yields one row
per pair, in row-major order, and keeps only a bounded result chunk inside
Memgraph:

```cypher
CALL stride_align.cdist(
  ['kitten', 'sitting'],
  ['kitten', 'bitten', 'smitten'],
  'jaro_winkler'
)
YIELD query_index, target_index, score
RETURN query_index, target_index, score;
```

Both indices are zero-based. The procedure accepts optional scorer parameters
after the scorer name:

```text
match_score, mismatch_score, gap_open_score, gap_extend_score,
prefix_weight, prefix_threshold, prefix_cap, chunk_size
```

The default `chunk_size` is 256 rows. It changes only the bounded handoff size,
not ordering or results. Downstream `WHERE`, `ORDER BY`, aggregation, and
`LIMIT` clauses consume the emitted rows normally; a stopped query cleans up
the stream before the next invocation.

## Test against Memgraph

The Python integration suite builds the module, extracts the public API header
from the pinned Memgraph image, starts a real server, and exercises scalar and
streaming calls:

```sh
pytest tests/memgraph
```

Set `STRIDE_ALIGN_MEMGRAPH_CONTAINER_RUNTIME` to `podman` or `docker`, and
override `STRIDE_ALIGN_MEMGRAPH_IMAGE` when testing another compatible image.
