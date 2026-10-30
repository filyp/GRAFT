#!/bin/bash
# neurons.py at step 990 on seeds $SEEDS (default 2-5) of the persona_qa_topic S=0 runs. Run from rl/.
for s in ${SEEDS:-2 3 4 5}; do
  r=$(ls -d output/fortho_topic_gr_t[ab]/*_fo0_s$s)
  .venv/bin/python ../selectivity/neurons.py --run_dir $r --out_dir ../selectivity/results/topic_neurons --steps ${STEPS:-990} \
    --n 256 > ../selectivity/logs/tp/neurons_$(basename $r).log 2>&1 || echo "NEURONS_FAILED $r"
  echo "NEURONS_DONE $r $(date)"
done
echo ALL_DONE
