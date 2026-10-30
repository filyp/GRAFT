#!/bin/bash
# monitored-word-only rankings ("beautiful", the routing monitor's word) on the 5 topic seeds: neurons.py
# (exact single ablations), one-shot gradient, greedy lam 0, all into results/topic_neurons_mon; then generation evals
# of the three rankings with --neurons_dir pointing there. From rl/.
ND=../selectivity/results/topic_neurons_mon
for s in 1 2 3 4 5; do
  r=$(ls -d output/fortho_topic_gr_t[ab]/*_fo0_s$s); b=$(basename $r)
  .venv/bin/python ../selectivity/neurons.py --run_dir $r --out_dir $ND --steps 990 --n 256 --words beautiful \
    > ../selectivity/logs/tp/neuronsmon_$b.log 2>&1 || echo "NEURONS_FAILED $r"
  PYTORCH_ALLOC_CONF=expandable_segments:True .venv/bin/python ../selectivity/neuron_greedy.py --run_dir $r \
    --pt $ND/$b/step990.pt --n_abl 0 --validate --chunks 2 > ../selectivity/logs/tp/grad0mon_$b.log 2>&1 || echo "GRAD0_FAILED $r"
  PYTORCH_ALLOC_CONF=expandable_segments:True .venv/bin/python ../selectivity/neuron_greedy.py --run_dir $r \
    --pt $ND/$b/step990.pt --lam 0 --n_abl 96 --n_sub 64 > ../selectivity/logs/tp/greedymon_$b.log 2>&1 || echo "GREEDY_FAILED $r"
  echo "SEED_DONE $r $(date)"
done
M=""; for rk in lowhack grad0 greedy0; do for k in 464 456 448 440 432 416; do M="$M,nk_keep_${rk}_$k"; done; done
M=${M#,}
for s in 1 2 3 4 5; do ls -d output/fortho_topic_gr_t[ab]/*_fo0_s$s; done | xargs -P 3 -I{} sh -c \
  'r={}; .venv/bin/python ../selectivity/eval_projection.py --run_dir $r --out ../selectivity/results/topic_mon/$(basename $r).json \
     --n_eval 256 --steps tail --neurons_dir '$ND' --modes '$M' > ../selectivity/logs/tp/topic_mon_$(basename $r).log 2>&1 || echo "EVAL_FAILED $r"'
echo "ALL_DONE $(date)"
