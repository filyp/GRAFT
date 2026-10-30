#!/bin/bash
# one-shot gradient ranking (greedy step 0) on the 5 topic seeds: validate against exact single ablation, then
# generation evals of nk_keep_grad0. From rl/.
for s in 1 2 3 4 5; do
  r=$(ls -d output/fortho_topic_gr_t[ab]/*_fo0_s$s)
  pt=../selectivity/results/topic_neurons/$(basename $r)/step990.pt
  log=../selectivity/logs/tp/grad0_$(basename $r).log
  PYTORCH_ALLOC_CONF=expandable_segments:True .venv/bin/python ../selectivity/neuron_greedy.py --run_dir $r --pt $pt \
    --n_abl 0 --validate > $log 2>&1 \
  || PYTORCH_ALLOC_CONF=expandable_segments:True .venv/bin/python ../selectivity/neuron_greedy.py --run_dir $r --pt $pt \
    --n_abl 0 --validate --n_sub 128 >> $log 2>&1 || echo "GRAD0_FAILED $r"
  echo "GRAD0_DONE $r $(date)"
done
MODES=nk_keep_grad0_464,nk_keep_grad0_456,nk_keep_grad0_448,nk_keep_grad0_440,nk_keep_grad0_432,nk_keep_grad0_416 \
  OUTD=topic_grad0 P=3 bash ../selectivity/topic_nk.sh
