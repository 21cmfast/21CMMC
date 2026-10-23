#!/usr/bin/env bash
# Does mpirun return after a master-rank exception? Compare a plain launch with
# `python -m mpi4py`, which calls MPI_Abort on an unhandled exception.
# Usage: hang_test.sh LOGDIR   (PYTHON = venv python)
LOG=$1; PY=${PYTHON:-python}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$LOG"
for mode in plain mpi4py; do
  if [ "$mode" = plain ]; then args=("$HERE/hang_repro.py"); else args=(-m mpi4py "$HERE/hang_repro.py"); fi
  start=$(date +%s)
  timeout 60 mpirun -n 4 "$PY" "${args[@]}" > "$LOG/hang_$mode.log" 2>&1
  echo "$mode: exit=$? after $(( $(date +%s) - start ))s (124 = killed by the 60s timeout)"
  pkill -f "hang_r""epro" 2>/dev/null; sleep 1
done
