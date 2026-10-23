#!/usr/bin/env bash
# Kill-and-resume test for one sampler under MPI.
#
# Usage: resume_test.sh SAMPLER OUTDIR KILL_AFTER_SECONDS [NPROC]
# Needs PYTHON pointing at the venv's python. TOY_SLEEP makes each evaluation slow.
set -u
SAMPLER=$1; OUT=$2; KILL_AFTER=$3; NPROC=${4:-4}
PY=${PYTHON:-python}
HERE=$(cd "$(dirname "$0")" && pwd)
export TOY_SLEEP=${TOY_SLEEP:-0.02}
rm -rf "$OUT"

echo "== first run, killed with SIGTERM after ${KILL_AFTER}s"
timeout -s TERM "$KILL_AFTER" mpirun -n "$NPROC" "$PY" "$HERE/run_cosmosis_toy.py" \
    --sampler "$SAMPLER" --out "$OUT" > "$OUT.run1.log" 2>&1
echo "exit=$? (124 means it was killed)"
sleep 2
pkill -f "run_cosmosis_toy.py --sampler $SAMPLER --out $OUT" && echo "(cleaned up stray ranks)"
grep -E "evaluations" "$OUT.run1.log" || echo "(no completion summary: run was interrupted)"
ls -la "$OUT"

echo "== second run with --resume"
mpirun -n "$NPROC" "$PY" "$HERE/run_cosmosis_toy.py" \
    --sampler "$SAMPLER" --out "$OUT" --resume > "$OUT.run2.log" 2>&1
echo "exit=$?"
grep -E "evaluations|WALL|esum" "$OUT.run2.log"

echo "== check"
"$PY" "$HERE/check_output.py" "$OUT/chain.txt"
