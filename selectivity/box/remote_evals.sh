#!/bin/bash
# BOX: post-hoc evals (tail checkpoints, n=256) of sweep $SW, each env as soon as its 3 runs are done, the 3 runs of
# an env in parallel. Results -> ~/GRAFT/selectivity/results/$OUTD. Log: ~/remote_evals.log
SW=${SW:?}; OUTD=${OUTD:?}; MODES=${MODES:-both,retain_only,proj_upgate}
cd ~/GRAFT/rl && source .venv/bin/activate
ev() {
  for r in output/$SW/$1*_s[0-9]; do
    python ../selectivity/eval_projection.py --run_dir $r --out ../selectivity/results/$OUTD/$(basename $r).json \
        --n_eval 256 --steps tail --modes $MODES > ~/eval_$(basename $r).log 2>&1 || echo "EVAL_FAILED $r" &
  done
  wait; echo "EVALS_DONE $1 $(date)"
}
until [ "$(grep -c "DONE .*persona_qa.* OK" ~/remote_run.log)" = "3" ]; do sleep 60; done; ev persona_qa
until grep -q SWEEP_EXIT ~/remote_run.log; do sleep 60; done; ev cities_qa
echo ALL_EVALS_DONE
