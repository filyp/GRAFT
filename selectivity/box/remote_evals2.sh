#!/bin/bash
# BOX: after sweep $SW exits, post-hoc evals (tail checkpoints, n=256, modes $MODES) of all its runs in parallel.
# Results -> ~/GRAFT/selectivity/results/$OUTD. Log: ~/remote_evals.log
SW=${SW:?}; OUTD=${OUTD:?}; MODES=${MODES:-both,retain_only}
until grep -q SWEEP_EXIT ~/remote_run.log 2>/dev/null; do sleep 60; done
cd ~/GRAFT/rl && source .venv/bin/activate   # after the sweep: on a fresh box remote_run.sh creates the venv
for r in output/$SW/*_s[0-9]; do
  python ../selectivity/eval_projection.py --run_dir $r --out ../selectivity/results/$OUTD/$(basename $r).json \
      --n_eval 256 --steps tail --modes $MODES > ~/eval_$(basename $r).log 2>&1 || echo "EVAL_FAILED $r" &
done
wait; echo "ALL_EVALS_DONE $(date)"
