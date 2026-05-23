"""Microbenchmark for the length-difference pruning in
``cdist_above_threshold`` and ``cdist_top_k``.

Generates random strings with a configurable length spread, then
sweeps thresholds (for ``cdist_above_threshold``) and k values (for
``cdist_top_k``), reporting wall-clock seconds and pairs-per-second
for each setting. The pruning is invisible at the API level — it
only matters how the timing changes as the threshold rises (more
pruning ⇒ less SIMD work ⇒ higher throughput).

Usage:
    tools/bench_cdist_pruning.py
    tools/bench_cdist_pruning.py --n 500 --m 600 --max-len 40 \
        --scorer JARO_WINKLER --cpu-count 8
"""

from __future__ import annotations

import argparse
import random
import string
import time

import stride_align as sa


def _rand_str(rng, n):
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(n))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=400,
                   help="number of queries")
    p.add_argument("--m", type=int, default=400,
                   help="number of targets")
    p.add_argument("--min-len", type=int, default=4)
    p.add_argument("--max-len", type=int, default=40)
    p.add_argument("--scorer", default="JARO",
                   help="Scorer enum name (JARO, JARO_WINKLER, "
                        "LEVENSHTEIN_NORMALIZED, DAMERAU_LEVENSHTEIN_NORMALIZED)")
    p.add_argument("--cpu-count", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.seed)
    qs = [_rand_str(rng, rng.randint(args.min_len, args.max_len))
          for _ in range(args.n)]
    ts = [_rand_str(rng, rng.randint(args.min_len, args.max_len))
          for _ in range(args.m)]
    scorer = getattr(sa.Scorer, args.scorer)
    total_pairs = args.n * args.m

    print(f"Setup: N={args.n} M={args.m} pairs={total_pairs} "
          f"length=[{args.min_len},{args.max_len}] scorer={args.scorer} "
          f"cpu_count={args.cpu_count}")
    print()

    print("cdist_above_threshold (sweeps threshold)")
    print(f"  {'threshold':>10}  {'seconds':>10}  {'pairs/s':>14}  "
          f"{'matches':>10}")
    for threshold in [0.0, 0.3, 0.5, 0.7, 0.85, 0.95, 0.99]:
        t0 = time.perf_counter()
        out = list(sa.cdist_above_threshold(
            qs, ts, scorer=scorer, threshold=threshold,
            cpu_count=args.cpu_count,
        ))
        elapsed = time.perf_counter() - t0
        rate = total_pairs / elapsed if elapsed > 0 else float("inf")
        print(f"  {threshold:>10.2f}  {elapsed:>10.4f}  {rate:>14,.0f}  "
              f"{len(out):>10}")
    print()

    print("cdist_top_k (sweeps k)")
    print(f"  {'k':>10}  {'seconds':>10}  {'pairs/s':>14}")
    for k in [1, 10, 100, 1000, 10000]:
        t0 = time.perf_counter()
        sa.cdist_top_k(qs, ts, scorer=scorer, k=k,
                       cpu_count=args.cpu_count)
        elapsed = time.perf_counter() - t0
        rate = total_pairs / elapsed if elapsed > 0 else float("inf")
        print(f"  {k:>10}  {elapsed:>10.4f}  {rate:>14,.0f}")


if __name__ == "__main__":
    main()
