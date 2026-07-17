#!/usr/bin/env bash
# =============================================================================
# lazy_f_soundness_hunter.sh
#
# WHAT THIS SEARCHES FOR
# ----------------------
# A concrete counter-example proving the *incumbent* striped-Farrar local
# Smith-Waterman kernel can return a WRONG (too-low) score.
#
# The incumbent kernel offered a "bounded" lazy-F correction that stopped
# propagating the F (horizontal-gap) wavefront down the striped segments as
# soon as  v_f <= h[segment] + gap.  That early-exit is only valid if h is
# monotonic non-increasing across the segments -- which is NOT true in general:
# a later segment can have a smaller h, so the carried F should still raise it.
# When that happens the bounded scan under-propagates and the reported local
# SW score comes out too low.  Silent -- wrong number, no crash.
# (See docs/known-issue-bounded-lazy-f-scan.md.)
#
# The kernel exposes both the buggy scan (--sw-farrar-i32-strategy bounded,
# bounded-u4) and a provably-correct reference (materialized -- the full
# unbounded scan).  For a given (sequence corpus, scoring) they must agree.
# ANY disagreement is a proof the bounded path is unsound.
#
# The bug is RARE: on the dev box, ~13k random inputs under default scoring
# produced zero disagreements.  It is most reachable where horizontal gaps are
# cheap relative to matches (large match, small gap) so the F wavefront
# travels across many segments before decaying -- that is the regime this
# hunter emphasises.  It runs one process per core, forever, until it finds
# enough counter-examples or you stop it.
#
# USAGE (on the target box, e.g. naamah):
#   nohup bash tools/lazy_f_soundness_hunter.sh > /tmp/lazyf_hunter.boot 2>&1 &
#   # watch findings:   tail -f /tmp/lazyf_hunt/findings.tsv
#   # watch progress:   tail -f /tmp/lazyf_hunt/progress.log
#   # stop:             touch /tmp/lazyf_hunt/STOP
#
# Env overrides: REPO, BACKEND (avx2|avx512bwvl|neon), JOBS, OUTDIR, MAX_FINDINGS
# =============================================================================
set -u

REPO="${REPO:-$HOME/dev/stride-align}"
BACKEND="${BACKEND:-avx2}"
JOBS="${JOBS:-$(nproc)}"
OUTDIR="${OUTDIR:-/tmp/lazyf_hunt}"
MAX_FINDINGS="${MAX_FINDINGS:-200}"
BASELINE_REF="${BASELINE_REF:-origin/cdist-exp-u64}"   # the incumbent (pre-fix) commit

mkdir -p "$OUTDIR"
FINDINGS="$OUTDIR/findings.tsv"
PROGRESS="$OUTDIR/progress.log"
STOP="$OUTDIR/STOP"
rm -f "$STOP"

log() { echo "$(date +%FT%T) $*" | tee -a "$PROGRESS"; }

# --- 1. Build the incumbent (bounded-capable) microbench in an isolated worktree
VENV="$OUTDIR/venv"
if [ ! -x "$VENV/bin/python" ]; then
  command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; }
  export PATH="$HOME/.local/bin:$PATH"
  uv venv "$VENV" >/dev/null 2>&1
  uv pip install --python "$VENV/bin/python" -q nanobind cmake ninja
fi
CMAKE="$VENV/bin/cmake"; [ -x "$CMAKE" ] || CMAKE=cmake
ND="$("$VENV/bin/python" -m nanobind --cmake_dir)"
WT="$OUTDIR/incumbent"
if [ ! -d "$WT" ]; then
  git -C "$REPO" fetch -q origin || true
  git -C "$REPO" -c core.hooksPath=/dev/null worktree add -f "$WT" "$BASELINE_REF"
fi
# x86 target on Intel/AMD, arm_neon target on ARM
TGT=stride_align_x86_microbench; [ "$BACKEND" = neon ] && TGT=stride_align_arm_neon_microbench
if ! ls "$WT"/build/perf/*microbench >/dev/null 2>&1; then
  log "building incumbent microbench ($TGT) ..."
  "$CMAKE" -S "$WT" -B "$WT/build/perf" -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DSTRIDE_ALIGN_BUILD_MICROBENCH=ON -DPython_EXECUTABLE="$VENV/bin/python" \
    -Dnanobind_DIR="$ND" > "$OUTDIR/cfg.log" 2>&1
  "$CMAKE" --build "$WT/build/perf" --target "$TGT" -j > "$OUTDIR/bld.log" 2>&1 \
    || { log "BUILD FAILED (see $OUTDIR/bld.log)"; exit 1; }
fi
BIN="$(find "$WT/build/perf" -name "$TGT" -type f | head -1)"
[ -x "$BIN" ] || { log "no microbench binary"; exit 1; }
log "hunter up: bin=$BIN backend=$BACKEND jobs=$JOBS findings->$FINDINGS"
: > "$FINDINGS"

# --- 2. One worker per core: force bounded/bounded-u4 vs materialized, forever
score() { "$BIN" $1 2>/dev/null | grep -oE 'score=[0-9-]+' | head -1 | cut -d= -f2; }

worker() {
  local wid="$1"
  local LENS=(200 384 512 768 1024 1536 2048 3072 4096)
  local WIDTHS=(16 32)
  local PASSES=(english chinese)
  # "match mismatch gap-open gap-extend" -- cheap gaps make F travel far
  local SCORING=("8 -9 -1 -1" "12 -13 -1 -1" "20 -21 -1 -1" "6 -7 -1 -1" \
                 "30 -31 -1 -1" "8 -9 -2 -1" "50 -51 -1 -1" "5 -6 -1 -1")
  local seed=$(( wid + 1 )) checked=0
  while [ ! -e "$STOP" ]; do
    for len in "${LENS[@]}"; do for w in "${WIDTHS[@]}"; do
      for pass in "${PASSES[@]}"; do for sc in "${SCORING[@]}"; do
        set -- $sc
        local base="--backend $BACKEND --variant sw-farrar-score --shape 1:many --pass $pass --width $w --length $len --seed $seed --match $1 --mismatch $2 --gap-open $3 --gap-extend $4 --iterations 1 --warmups 0"
        local ref; ref=$(score "$base --sw-farrar-i32-strategy materialized")
        [ -z "$ref" ] && continue
        local strat
        for strat in bounded bounded-u4; do
          local got; got=$(score "$base --sw-farrar-i32-strategy $strat")
          checked=$((checked+1))
          if [ -n "$got" ] && [ "$got" != "$ref" ]; then
            printf '%s\tworker=%s\tstrategy=%s\tbounded=%s\tmaterialized=%s\tdelta=%s\tREPRO: %s --sw-farrar-i32-strategy %s\n' \
              "$(date +%FT%T)" "$wid" "$strat" "$got" "$ref" "$((ref-got))" "$base" "$strat" >> "$FINDINGS"
            if [ "$(grep -c . "$FINDINGS")" -ge "$MAX_FINDINGS" ]; then touch "$STOP"; fi
          fi
        done
        [ -e "$STOP" ] && return
      done; done
    done; done
    seed=$(( seed + JOBS ))
    [ "$wid" -eq 0 ] && log "progress: seed~$seed, worker0 checked~$checked, findings=$(grep -c . "$FINDINGS" 2>/dev/null || echo 0)"
  done
}

for wid in $(seq 0 $((JOBS-1))); do worker "$wid" & done
wait
log "hunter stopped. total findings: $(grep -c . "$FINDINGS" 2>/dev/null || echo 0)"
