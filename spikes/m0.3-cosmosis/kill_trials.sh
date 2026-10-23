#!/usr/bin/env bash
# How often does a SIGTERM leave nautilus's checkpoint unusable?
# For each kill time, run CosmoSIS+nautilus under MPI, kill it, then try to
# construct a nautilus Sampler with resume=True on the checkpoint (what CosmoSIS does).
#
# Usage: kill_trials.sh OUTROOT "15 25 35 ..." [NPROC]
set -u
ROOT=$1; TIMES=$2; NPROC=${3:-4}
PY=${PYTHON:-python}
HERE=$(cd "$(dirname "$0")" && pwd)
export TOY_SLEEP=${TOY_SLEEP:-0.02}
mkdir -p "$ROOT"
for t in $TIMES; do
  OUT="$ROOT/kill_$t"; rm -rf "$OUT"
  timeout -s TERM "$t" mpirun -n "$NPROC" "$PY" "$HERE/run_cosmosis_toy.py" \
      --sampler nautilus --out "$OUT" > "$OUT.log" 2>&1
  sleep 2; pkill -f "run_cosmosis_toy.py --sampler nautilus --out $OUT"
  "$PY" - "$OUT/chain.sampler_status.hdf5" "$t" <<'EOF'
import sys
from pathlib import Path
import numpy as np
from nautilus import Sampler
path, t = sys.argv[1], sys.argv[2]
if not Path(path).exists():
    print(f"kill@{t}s: no checkpoint yet"); sys.exit()
try:
    s = Sampler(lambda u: u, lambda x: 0.0, n_dim=5, filepath=path, resume=True)
    print(f"kill@{t}s: checkpoint OK ({len(s.bounds)} bounds)")
except Exception as e:
    print(f"kill@{t}s: CORRUPT ({type(e).__name__}: {str(e)[:60]})")
EOF
done
