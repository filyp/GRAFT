#!/bin/bash
# neuron_greedy.py (lam $LAM, 96 ablations, 64-prompt subset) on the step-990 results of the topic S=0 seeds. From rl/.
for s in ${SEEDS:-1 2 3 4 5}; do
  r=$(ls -d ${RUNGLOB:-output/fortho_topic_gr_t[ab]}/*_fo0_s$s)
  PYTORCH_ALLOC_CONF=expandable_segments:True .venv/bin/python ../selectivity/neuron_greedy.py --run_dir $r \
    --pt ../selectivity/results/${ND:-topic_neurons}/$(basename $r)/step990.pt --lam ${LAM:-0} --n_abl 96 --n_sub 64 \
    > ../selectivity/logs/tp/greedy${LAM:-0}_$(basename $r).log 2>&1 || echo "GREEDY_FAILED $r"
  echo "GREEDY_DONE $r $(date)"
done
echo ALL_DONE
