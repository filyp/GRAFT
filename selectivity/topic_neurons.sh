#!/bin/bash
# neurons.py on the persona_qa_topic S=0 runs (reports/2026.10.29_topic_pareto.md step 2). Run from rl/.
# STEPS_S1 for seed 1 (ranking stability across checkpoints), STEPS for the others. One process (GPU-bound).
OUTD=${OUTD:-../selectivity/results/topic_neurons}
for r in $(ls -d output/fortho_topic_gr_t[ab]/*_fo0_s[0-9]); do
  st=${STEPS:-990}; [[ $r == *_s1 ]] && st=${STEPS_S1:-910,990}
  .venv/bin/python ../selectivity/neurons.py --run_dir $r --out_dir $OUTD --steps $st --n ${N:-256} \
    > ../selectivity/logs/tp/neurons_$(basename $r).log 2>&1 || echo "NEURONS_FAILED $r"
  echo "NEURONS_DONE $r $(date)"
done
echo ALL_DONE
