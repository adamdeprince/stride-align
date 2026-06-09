#!/usr/bin/env python3
"""Cross-architecture rapidfuzz-vs-shim benchmark.

Times the most common rapidfuzz entry points against
``stride_align.rapidfuzz`` (the shim) on a fixed reproducible corpus,
detects which SIMD backend stride-align is using, and emits a single-
line JSON record + a markdown row suitable for stitching into
``BENCHMARK.md``.

Run on each cross-arch SSH host:

    .venv/bin/python tools/refresh_rapidfuzz_benchmarks.py \\
        --output bench-$(hostname)-$(date +%Y%m%d).json

Then combine the JSON outputs into a single markdown table with
``python tools/refresh_rapidfuzz_benchmarks.py --combine bench-*.json``.

The benchmark corpus is generated from a deterministic seed so the
numbers are comparable across hosts. The workloads are picked to
exercise the surfaces where stride-align should be most competitive:

* ``fuzz.ratio`` — the most common rapidfuzz scorer; tests bit-parallel
  Indel throughput.
* ``fuzz.WRatio`` — composite scorer; tests the shim's reimplemented
  recipe.
* ``distance.Levenshtein.distance`` — bit-parallel Myers / Hyyrö
  Levenshtein.
* ``distance.Indel.normalized_similarity`` with ``score_cutoff`` —
  kernel-level early-exit on Indel.
* ``process.extract`` from a 10k-choice corpus — exercises the
  ``sa.*_top_k`` length-pruning + adaptive-bound fast path.
* ``process.cdist`` 50×100 — exercises the multi-threaded C++ kernel
  on the cdist fast path.

We expect the shim to be roughly competitive with upstream rapidfuzz
on x86 (where rapidfuzz has the most tuning), and substantially
faster on Loongson LASX / RISC-V RVV / AVX-512BWVL (where rapidfuzz
doesn't ship SIMD).
"""

from __future__ import annotations

import argparse
import datetime
import importlib
import json
import platform
import random
import sys
import time
from typing import Any, Callable


def _detect_backend() -> str:
    """Return the stride-align backend module name in use, e.g.
    ``_avx512bwvl`` or ``_neon``."""
    sa = importlib.import_module("stride_align")
    backend = getattr(sa, "_LEVENSHTEIN_BACKEND", None)
    if backend is None:
        return "<unknown>"
    return str(getattr(backend, "__name__", "<unknown>")).split(".")[-1]


def _make_corpus(seed: int = 0xC0FFEE) -> dict:
    """Generate a deterministic corpus."""
    rng = random.Random(seed)
    alphabet = "abcdefghijklmnopqrstuvwxyz "
    short_pairs = []
    for _ in range(5000):
        n_a = rng.randint(5, 25)
        n_b = rng.randint(5, 25)
        short_pairs.append((
            "".join(rng.choices(alphabet, k=n_a)),
            "".join(rng.choices(alphabet, k=n_b)),
        ))
    medium_pairs = []
    for _ in range(1000):
        n_a = rng.randint(30, 80)
        n_b = rng.randint(30, 80)
        medium_pairs.append((
            "".join(rng.choices(alphabet, k=n_a)),
            "".join(rng.choices(alphabet, k=n_b)),
        ))
    # Long pairs exercise the multi-word bit-parallel Indel kernel at
    # K = 2 through K = 8 (patterns 65..512 chars). Each band sized so
    # the per-band total walltime is on the same order as the short /
    # medium workloads above; fewer pairs at higher K because each
    # call is more work.
    long_bands = {
        "K2_long_pairs": (65, 128, 600),    # K = 2
        "K3_long_pairs": (129, 192, 500),   # K = 3
        "K4_long_pairs": (193, 256, 400),   # K = 4
        "K5_long_pairs": (257, 320, 300),   # K = 5
        "K6_long_pairs": (321, 384, 250),   # K = 6
        "K7_long_pairs": (385, 448, 200),   # K = 7
        "K8_long_pairs": (449, 512, 150),   # K = 8
    }
    long_pairs_by_band = {}
    for label, (lo, hi, count) in long_bands.items():
        band_pairs = []
        for _ in range(count):
            n_a = rng.randint(lo, hi)
            n_b = rng.randint(lo, hi)
            band_pairs.append((
                "".join(rng.choices(alphabet, k=n_a)),
                "".join(rng.choices(alphabet, k=n_b)),
            ))
        long_pairs_by_band[label] = band_pairs
    corpus_10k = []
    for _ in range(10_000):
        n = rng.randint(8, 40)
        corpus_10k.append("".join(rng.choices(alphabet, k=n)))
    cdist_queries = []
    cdist_choices = []
    for _ in range(50):
        cdist_queries.append("".join(rng.choices(alphabet, k=rng.randint(10, 30))))
    for _ in range(100):
        cdist_choices.append("".join(rng.choices(alphabet, k=rng.randint(10, 30))))

    return {
        "short_pairs":  short_pairs,
        "medium_pairs": medium_pairs,
        "extract_query":  "hello world test",
        "extract_corpus": corpus_10k,
        "cdist_queries":  cdist_queries,
        "cdist_choices":  cdist_choices,
        **long_pairs_by_band,
    }


def _time(fn: Callable, *, iters: int = 5, warmup: int = 2) -> float:
    """Return min-of-5 elapsed ms per invocation-group."""
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(5):
        t0 = time.perf_counter_ns()
        for _ in range(iters):
            fn()
        elapsed = (time.perf_counter_ns() - t0) / iters / 1e6
        if elapsed < best:
            best = elapsed
    return best


def _bench_pair_loop(fn: Callable, pairs: list, iters: int = 3) -> float:
    return _time(lambda: [fn(a, b) for a, b in pairs], iters=iters)


def run_benchmarks(corpus: dict) -> dict:
    upstream_rf = importlib.import_module("rapidfuzz")
    upstream_fuzz = upstream_rf.fuzz
    upstream_dist = upstream_rf.distance
    upstream_proc = upstream_rf.process

    sa_rf = importlib.import_module("stride_align.rapidfuzz")
    shim_fuzz = sa_rf.fuzz
    shim_dist = sa_rf.distance
    shim_proc = sa_rf.process

    short_pairs = corpus["short_pairs"]
    medium_pairs = corpus["medium_pairs"]
    eq = corpus["extract_query"]
    ec = corpus["extract_corpus"]
    cq = corpus["cdist_queries"]
    cc = corpus["cdist_choices"]

    results = {}

    print("  fuzz.ratio (5000 short pairs)...", file=sys.stderr)
    up_t = _bench_pair_loop(upstream_fuzz.ratio, short_pairs)
    sh_t = _bench_pair_loop(shim_fuzz.ratio,     short_pairs)
    results["fuzz.ratio_short_5000"] = {"upstream_ms": up_t, "shim_ms": sh_t,
                                         "ratio": sh_t / up_t}

    print("  fuzz.ratio (1000 medium pairs)...", file=sys.stderr)
    up_t = _bench_pair_loop(upstream_fuzz.ratio, medium_pairs)
    sh_t = _bench_pair_loop(shim_fuzz.ratio,     medium_pairs)
    results["fuzz.ratio_medium_1000"] = {"upstream_ms": up_t, "shim_ms": sh_t,
                                          "ratio": sh_t / up_t}

    print("  fuzz.WRatio (1000 medium pairs)...", file=sys.stderr)
    up_t = _bench_pair_loop(upstream_fuzz.WRatio, medium_pairs)
    sh_t = _bench_pair_loop(shim_fuzz.WRatio,     medium_pairs)
    results["fuzz.WRatio_medium_1000"] = {"upstream_ms": up_t, "shim_ms": sh_t,
                                           "ratio": sh_t / up_t}

    print("  Levenshtein.distance (5000 short pairs)...", file=sys.stderr)
    up_t = _bench_pair_loop(upstream_dist.Levenshtein.distance, short_pairs)
    sh_t = _bench_pair_loop(shim_dist.Levenshtein.distance,     short_pairs)
    results["Levenshtein.distance_short_5000"] = {"upstream_ms": up_t,
                                                    "shim_ms": sh_t,
                                                    "ratio": sh_t / up_t}

    # Long-pair Levenshtein sweep across K = 2..4. Exercises the
    # multi-word bit-parallel Myers/Hyyrö kernel's hand-specialised K
    # bands (m = 65..256). Same long_pairs corpus as the Indel sweep
    # so the call counts line up across the two algorithms.
    for k_label in ("K2", "K3", "K4"):
        key = f"{k_label}_long_pairs"
        if key not in corpus:
            continue
        pairs = corpus[key]
        print(f"  Levenshtein.distance {k_label} long pairs ({len(pairs)})...", file=sys.stderr)
        up_t = _bench_pair_loop(upstream_dist.Levenshtein.distance, pairs)
        sh_t = _bench_pair_loop(shim_dist.Levenshtein.distance,     pairs)
        results[f"Levenshtein.distance_{k_label}_long"] = {"upstream_ms": up_t,
                                                            "shim_ms": sh_t,
                                                            "ratio": sh_t / up_t}

    print("  Indel.normalized_similarity score_cutoff=0.7 (5000 short pairs)...", file=sys.stderr)
    up_t = _bench_pair_loop(
        lambda a, b: upstream_dist.Indel.normalized_similarity(a, b, score_cutoff=0.7),
        short_pairs)
    sh_t = _bench_pair_loop(
        lambda a, b: shim_dist.Indel.normalized_similarity(a, b, score_cutoff=0.7),
        short_pairs)
    results["Indel.normalized_similarity_cutoff_5000"] = {"upstream_ms": up_t,
                                                            "shim_ms": sh_t,
                                                            "ratio": sh_t / up_t}

    # Long-pair Indel sweep across K = 2..8. Exercises the multi-word
    # bit-parallel kernel's hand-specialised K bands (m in 65..512)
    # where rapidfuzz historically had a wide lead on stride-align.
    for k_label in ("K2", "K3", "K4", "K5", "K6", "K7", "K8"):
        key = f"{k_label}_long_pairs"
        if key not in corpus:
            continue
        pairs = corpus[key]
        print(f"  Indel.distance {k_label} long pairs ({len(pairs)})...", file=sys.stderr)
        up_t = _bench_pair_loop(upstream_dist.Indel.distance, pairs)
        sh_t = _bench_pair_loop(shim_dist.Indel.distance,     pairs)
        results[f"Indel.distance_{k_label}_long"] = {"upstream_ms": up_t,
                                                       "shim_ms": sh_t,
                                                       "ratio": sh_t / up_t}

    print("  process.extract limit=5 (10k corpus)...", file=sys.stderr)
    up_t = _time(lambda: upstream_proc.extract(eq, ec, scorer=upstream_fuzz.ratio, limit=5), iters=10)
    sh_t = _time(lambda: shim_proc.extract(eq, ec, scorer=shim_fuzz.ratio,         limit=5), iters=10)
    results["process.extract_10k"] = {"upstream_ms": up_t, "shim_ms": sh_t,
                                       "ratio": sh_t / up_t}

    print("  process.cdist 50x100 (fuzz.ratio)...", file=sys.stderr)
    up_t = _time(lambda: upstream_proc.cdist(cq, cc, scorer=upstream_fuzz.ratio), iters=10)
    sh_t = _time(lambda: shim_proc.cdist(cq, cc, scorer=shim_fuzz.ratio),         iters=10)
    results["process.cdist_50x100"] = {"upstream_ms": up_t, "shim_ms": sh_t,
                                        "ratio": sh_t / up_t}

    print("  process.cdist 50x100 (Levenshtein.distance)...", file=sys.stderr)
    up_t = _time(lambda: upstream_proc.cdist(cq, cc, scorer=upstream_dist.Levenshtein.distance), iters=10)
    sh_t = _time(lambda: shim_proc.cdist(cq, cc, scorer=shim_dist.Levenshtein.distance),         iters=10)
    results["process.cdist_50x100_lev"] = {"upstream_ms": up_t, "shim_ms": sh_t,
                                            "ratio": sh_t / up_t}

    return results


def _emit_markdown_row(host: str, backend: str, timings: dict) -> str:
    row = [host, backend]
    keys = sorted(timings.keys())
    for k in keys:
        ratio = timings[k]["ratio"]
        speedup_vs_upstream = 1.0 / ratio
        if speedup_vs_upstream >= 1.0:
            row.append(f"**{speedup_vs_upstream:.2f}×** ✓")
        else:
            row.append(f"{speedup_vs_upstream:.2f}×")
    return "| " + " | ".join(row) + " |"


def _emit_markdown_header(timings: dict) -> str:
    keys = sorted(timings.keys())
    cols = ["host", "backend"] + keys
    sep  = ["---"] * len(cols)
    return ("| " + " | ".join(cols) + " |\n"
            + "| " + " | ".join(sep) + " |")


def _combine_json(paths: list[str]) -> None:
    all_records = []
    for path in paths:
        with open(path) as f:
            all_records.append(json.load(f))
    if not all_records:
        return
    print(_emit_markdown_header(all_records[0]["timings"]))
    for rec in all_records:
        print(_emit_markdown_row(
            rec.get("host", "?"),
            rec.get("backend", "?"),
            rec["timings"],
        ))
    print()
    print("Values are **shim speedup vs upstream rapidfuzz**, formatted as")
    print("`Nx` (>= 1.0× = shim is faster, in bold; < 1.0× = upstream is faster).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", help="JSON output path")
    parser.add_argument("--combine", nargs="+",
                        help="Combine multiple JSON outputs into a markdown table")
    args = parser.parse_args()

    if args.combine:
        _combine_json(args.combine)
        return

    print("Building corpus...", file=sys.stderr)
    corpus = _make_corpus()
    backend = _detect_backend()
    host = platform.node()
    arch = platform.machine()
    print(f"Host: {host}  Arch: {arch}  Backend: {backend}", file=sys.stderr)
    print("Running benchmarks (each is a min-of-5)...", file=sys.stderr)
    timings = run_benchmarks(corpus)
    record = {
        "host": host,
        "arch": arch,
        "backend": backend,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "timings": timings,
    }
    if args.output:
        with open(args.output, "w") as f:
            json.dump(record, f, indent=2)
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        json.dump(record, sys.stdout, indent=2)
        print()


if __name__ == "__main__":
    main()
