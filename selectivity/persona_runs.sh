#!/bin/bash
# original persona_qa S=0 runs (fortho_persona_gr_{h,t}, seeds 1-6): eval_projection with $MODES -> results/$OUTD. Run from rl/.
OUTD=${OUTD:?}; P=${P:-2}; ND=${NEURONS_DIR:-../selectivity/results/persona_neurons}
ls -d output/fortho_persona_gr_[ht]/*_fo0_s[0-9] | xargs -P $P -I{} sh -c \
  'r={}; .venv/bin/python ../selectivity/eval_projection.py --run_dir $r --out ../selectivity/results/'$OUTD'/$(basename $r).json \
     --n_eval 256 --steps tail --neurons_dir '$ND' --modes '${MODES:?}' > ../selectivity/logs/tp/'$OUTD'_$(basename $r).log 2>&1 || echo "EVAL_FAILED $r"'
echo "ALL_DONE $(date)"
