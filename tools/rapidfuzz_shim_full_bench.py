#!/usr/bin/env python3
"""Comprehensive ``stride_align.rapidfuzz`` shim vs upstream rapidfuzz
benchmark. Walks every public entry point in the shim's ``fuzz``,
``distance`` and ``process`` namespaces, times each against the
upstream equivalent on a deterministic corpus, and emits a markdown
table plus a JSON record.

Use the smaller ``tools/refresh_rapidfuzz_benchmarks.py`` for the
cross-arch regression sweep; this one is for the comprehensive
ship-readiness comparison.

Run:

    .venv/bin/python tools/rapidfuzz_shim_full_bench.py \\
        --output bench-shim-full-$(date +%Y%m%d).json \\
        --markdown bench-shim-full-$(date +%Y%m%d).md
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
from collections.abc import Callable


def _make_corpus(seed: int = 0xC0FFEE) -> dict:
    """Deterministic per-band corpus."""
    rng = random.Random(seed)
    alphabet = "abcdefghijklmnopqrstuvwxyz "

    def pairs(n: int, mlo: int, mhi: int) -> list:
        return [
            ("".join(rng.choices(alphabet, k=rng.randint(mlo, mhi))),
             "".join(rng.choices(alphabet, k=rng.randint(mlo, mhi))))
            for _ in range(n)
        ]

    # Equal-length corpus for Hamming (which requires |s1| == |s2| with
    # pad=False; pad=True converts to length-difference mismatch cost).
    def equal_pairs(n: int, m: int) -> list:
        return [
            ("".join(rng.choices(alphabet, k=m)),
             "".join(rng.choices(alphabet, k=m)))
            for _ in range(n)
        ]

    return {
        "short":           pairs(5000, 5, 25),
        "medium":          pairs(1000, 30, 80),
        "long_K2":         pairs(500, 65, 128),
        "long_K3":         pairs(400, 129, 192),
        "long_K4":         pairs(300, 193, 256),
        "equal_short":     equal_pairs(2000, 20),
        "equal_medium":    equal_pairs(500, 60),
        "extract_query":   "hello world test",
        "extract_corpus": ["".join(rng.choices(alphabet, k=rng.randint(8, 40)))
                            for _ in range(10_000)],
        "cdist_queries":  ["".join(rng.choices(alphabet, k=rng.randint(10, 30)))
                            for _ in range(50)],
        "cdist_choices":  ["".join(rng.choices(alphabet, k=rng.randint(10, 30)))
                            for _ in range(100)],
    }


def _time_pair_loop(fn: Callable, pairs: list, iters: int = 3) -> float:
    """Min-of-5 wall time per loop, in ms."""
    for _ in range(2):
        [fn(a, b) for a, b in pairs]
    best = float("inf")
    for _ in range(5):
        t0 = time.perf_counter_ns()
        for _ in range(iters):
            [fn(a, b) for a, b in pairs]
        elapsed = (time.perf_counter_ns() - t0) / iters / 1e6
        if elapsed < best:
            best = elapsed
    return best


def _time_call(fn: Callable, iters: int = 10) -> float:
    for _ in range(2):
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


def _detect_backend() -> str:
    sa = importlib.import_module("stride_align")
    backend = getattr(sa, "_LEVENSHTEIN_BACKEND", None)
    if backend is None:
        return "<unknown>"
    return str(getattr(backend, "__name__", "<unknown>")).split(".")[-1]


def run_benchmarks(corpus: dict) -> dict:
    up_rf = importlib.import_module("rapidfuzz")
    up_fuzz = up_rf.fuzz
    up_dist = up_rf.distance
    up_proc = up_rf.process

    sa_rf = importlib.import_module("stride_align.rapidfuzz")
    sh_fuzz = sa_rf.fuzz
    sh_dist = sa_rf.distance
    sh_proc = sa_rf.process

    results: dict = {}

    def record(name: str, sh_fn, up_fn, pairs):
        sh = _time_pair_loop(sh_fn, pairs)
        up = _time_pair_loop(up_fn, pairs)
        results[name] = {"shim_ms": sh, "upstream_ms": up,
                         "ratio": sh / up if up > 0 else float("inf"),
                         "n_pairs": len(pairs)}
        print(f"  {name:<58s} shim {sh:>7.3f} ms  rf {up:>7.3f} ms  "
              f"ratio={sh/up:.2f}×", file=sys.stderr)

    # -----------------------------------------------------------------
    # fuzz.* (all 10 entry points)
    # -----------------------------------------------------------------
    print("# fuzz.* family", file=sys.stderr)
    for band_name, pairs in [("short", corpus["short"]),
                              ("medium", corpus["medium"])]:
        for fn_name in ("ratio", "partial_ratio",
                         "token_sort_ratio", "token_set_ratio", "token_ratio",
                         "partial_token_sort_ratio", "partial_token_set_ratio",
                         "partial_token_ratio",
                         "WRatio", "QRatio"):
            record(f"fuzz.{fn_name}_{band_name}",
                   getattr(sh_fuzz, fn_name), getattr(up_fuzz, fn_name), pairs)

    # -----------------------------------------------------------------
    # distance.* class methods (each class × 4 methods × 2 bands)
    # -----------------------------------------------------------------
    print("# distance.* classes", file=sys.stderr)
    distance_classes = [
        ("Indel",              sh_dist.Indel,              up_dist.Indel),
        ("Levenshtein",        sh_dist.Levenshtein,        up_dist.Levenshtein),
        ("OSA",                sh_dist.OSA,                up_dist.OSA),
        ("DamerauLevenshtein", sh_dist.DamerauLevenshtein, up_dist.DamerauLevenshtein),
        ("Jaro",               sh_dist.Jaro,               up_dist.Jaro),
        ("JaroWinkler",        sh_dist.JaroWinkler,        up_dist.JaroWinkler),
        ("LCSseq",             sh_dist.LCSseq,             up_dist.LCSseq),
    ]
    for cls_name, sh_cls, up_cls in distance_classes:
        for method_name in ("distance", "similarity",
                             "normalized_distance", "normalized_similarity"):
            sh_fn = getattr(sh_cls, method_name, None)
            up_fn = getattr(up_cls, method_name, None)
            if sh_fn is None or up_fn is None:
                continue
            for band_name, pairs in [("short", corpus["short"]),
                                      ("medium", corpus["medium"])]:
                record(f"distance.{cls_name}.{method_name}_{band_name}",
                       sh_fn, up_fn, pairs)

    # Hamming separately (requires equal lengths on the strict path).
    print("# distance.Hamming (equal-length corpus)", file=sys.stderr)
    for cls_name, sh_cls, up_cls in [("Hamming", sh_dist.Hamming, up_dist.Hamming)]:
        for method_name in ("distance", "similarity",
                             "normalized_distance", "normalized_similarity"):
            for band_name, pairs in [("equal_short", corpus["equal_short"]),
                                      ("equal_medium", corpus["equal_medium"])]:
                record(f"distance.{cls_name}.{method_name}_{band_name}",
                       getattr(sh_cls, method_name),
                       getattr(up_cls, method_name), pairs)

    # -----------------------------------------------------------------
    # K-band long pairs (Indel / Levenshtein / OSA)
    # -----------------------------------------------------------------
    print("# Long-pair K bands", file=sys.stderr)
    for k_label, k_key in [("K2", "long_K2"), ("K3", "long_K3"), ("K4", "long_K4")]:
        pairs = corpus[k_key]
        for cls_name, sh_cls, up_cls in distance_classes:
            if cls_name in ("Jaro", "JaroWinkler"):
                continue  # not K-banded
            record(f"distance.{cls_name}.distance_{k_label}",
                   sh_cls.distance, up_cls.distance, pairs)

    # -----------------------------------------------------------------
    # process.cdist with different scorers
    # -----------------------------------------------------------------
    print("# process.cdist 50x100", file=sys.stderr)
    cq, cc = corpus["cdist_queries"], corpus["cdist_choices"]
    for scorer_name, sh_s, up_s in [
        ("fuzz.ratio",            sh_fuzz.ratio,            up_fuzz.ratio),
        ("Levenshtein.distance",  sh_dist.Levenshtein.distance,  up_dist.Levenshtein.distance),
        ("Indel.distance",        sh_dist.Indel.distance,        up_dist.Indel.distance),
        ("OSA.distance",          sh_dist.OSA.distance,          up_dist.OSA.distance),
        ("Jaro.similarity",       sh_dist.Jaro.similarity,       up_dist.Jaro.similarity),
    ]:
        sh = _time_call(lambda: sh_proc.cdist(cq, cc, scorer=sh_s), iters=10)
        up = _time_call(lambda: up_proc.cdist(cq, cc, scorer=up_s), iters=10)
        name = f"process.cdist_50x100_{scorer_name.replace('.', '_')}"
        results[name] = {"shim_ms": sh, "upstream_ms": up,
                         "ratio": sh / up if up > 0 else float("inf"),
                         "n_pairs": 50 * 100}
        print(f"  {name:<58s} shim {sh:>7.3f} ms  rf {up:>7.3f} ms  "
              f"ratio={sh/up:.2f}×", file=sys.stderr)

    # -----------------------------------------------------------------
    # process.extract limit=5 from a 10k corpus
    # -----------------------------------------------------------------
    print("# process.extract limit=5 from 10k corpus", file=sys.stderr)
    eq, ec = corpus["extract_query"], corpus["extract_corpus"]
    for scorer_name, sh_s, up_s in [
        ("fuzz.ratio",            sh_fuzz.ratio,            up_fuzz.ratio),
        ("fuzz.WRatio",           sh_fuzz.WRatio,           up_fuzz.WRatio),
        ("Levenshtein.distance",  sh_dist.Levenshtein.distance,  up_dist.Levenshtein.distance),
        ("Indel.distance",        sh_dist.Indel.distance,        up_dist.Indel.distance),
    ]:
        sh = _time_call(lambda: sh_proc.extract(eq, ec, scorer=sh_s, limit=5), iters=10)
        up = _time_call(lambda: up_proc.extract(eq, ec, scorer=up_s, limit=5), iters=10)
        name = f"process.extract_10k_{scorer_name.replace('.', '_')}"
        results[name] = {"shim_ms": sh, "upstream_ms": up,
                         "ratio": sh / up if up > 0 else float("inf"),
                         "n_pairs": 10_000}
        print(f"  {name:<58s} shim {sh:>7.3f} ms  rf {up:>7.3f} ms  "
              f"ratio={sh/up:.2f}×", file=sys.stderr)

    return results


def emit_markdown(record: dict, path: str) -> None:
    timings = record["timings"]
    # Group by leading category for the markdown layout.
    sections: dict[str, list] = {}
    for name, v in timings.items():
        cat = name.split(".")[0]
        sections.setdefault(cat, []).append((name, v))

    with open(path, "w") as f:
        f.write(f"# stride-align rapidfuzz shim — full bench\n\n")
        f.write(f"- Host: `{record['host']}`\n")
        f.write(f"- Arch: `{record['arch']}`\n")
        f.write(f"- Backend: `{record['backend']}`\n")
        f.write(f"- Timestamp (UTC): {record['timestamp']}\n")
        f.write(f"- rapidfuzz: {record.get('rapidfuzz_version', '?')}\n\n")
        f.write("Lower ``ratio`` = shim faster than upstream rapidfuzz. ")
        f.write("Min-of-5 wall time, iters per measurement vary by workload.\n\n")

        # Summary count of wins / losses
        wins = sum(1 for _, v in timings.items() if v["ratio"] <= 1.05)
        ties = sum(1 for _, v in timings.items() if 1.05 < v["ratio"] <= 1.15)
        loses = sum(1 for _, v in timings.items() if v["ratio"] > 1.15)
        f.write(f"**Summary:** {wins} wins / {ties} ties / {loses} losses "
                f"(out of {len(timings)} benchmarks).\n\n")

        for cat, items in sections.items():
            f.write(f"## `{cat}`\n\n")
            f.write("| workload | shim ms | upstream ms | ratio | verdict |\n")
            f.write("|---|---:|---:|---:|---|\n")
            for name, v in sorted(items):
                r = v["ratio"]
                if r <= 0.85:
                    verdict = "shim wins"
                elif r <= 1.05:
                    verdict = "parity"
                elif r <= 1.5:
                    verdict = "close"
                else:
                    verdict = "**upstream**"
                f.write(f"| {name} | {v['shim_ms']:.3f} | "
                        f"{v['upstream_ms']:.3f} | {r:.2f}× | {verdict} |\n")
            f.write("\n")


def main(argv: list) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True, help="JSON output path")
    p.add_argument("--markdown", default=None, help="Markdown output path")
    p.add_argument("--seed", type=int, default=0xC0FFEE)
    args = p.parse_args(argv[1:])

    import rapidfuzz
    print(f"Host: {platform.node()}  Arch: {platform.machine()}  "
          f"Backend: {_detect_backend()}  rapidfuzz: {rapidfuzz.__version__}",
          file=sys.stderr)

    corpus = _make_corpus(seed=args.seed)
    timings = run_benchmarks(corpus)

    record = {
        "host":      platform.node(),
        "arch":      platform.machine(),
        "backend":   _detect_backend(),
        "platform":  platform.platform(),
        "python":    sys.version.split()[0],
        "rapidfuzz_version": rapidfuzz.__version__,
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "seed":      args.seed,
        "timings":   timings,
    }
    with open(args.output, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\nWrote {args.output}", file=sys.stderr)
    if args.markdown:
        emit_markdown(record, args.markdown)
        print(f"Wrote {args.markdown}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
