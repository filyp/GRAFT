#!/bin/bash
# Own variants (mix<f>, projx<k>) on the tail checkpoints of every run matching $1; results -> results/posthoc_tail_var
cd /srv/projects/GRAFT/rl && source .venv/bin/activate
MODES=${MODES:-mix0.1,mix0.25,mix0.5,projx1.5,projx2,projx3}
OUT=${OUT:-posthoc_tail_var}
for r in output/repro_persona_cities_gr/$1*_s[0-9]; do
  out=../selectivity/results/$OUT/$(basename $r).json
  [ -f $out ] && continue
  python ../selectivity/eval_projection.py --run_dir $r --out $out --n_eval 256 --steps tail --modes $MODES \
      > ../selectivity/logs/${OUT}_$(basename $r).log 2>&1 || echo "VAR_FAILED $r"
  echo "VAR_EVALUATED $(basename $r) $(date)"
done
