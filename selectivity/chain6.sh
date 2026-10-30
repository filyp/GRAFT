#!/bin/bash
# smaller-FORGET runs (m16f8: 240 FORGET neurons, m16f4: 120): neurons.py at step 990, then neuron-ablation evals
# (hack-ranked and norm-ranked, 8-48 neurons ablated) on the 5 seeds each. Run from rl/.
cd /srv/projects/GRAFT/rl
for f in 8 4; do b=$([ $f = 8 ] && echo ti || echo th)
  for r in $(ls -d output/fortho_topic_gr_$b/*_s[0-9]); do
    .venv/bin/python ../selectivity/neurons.py --run_dir $r --out_dir ../selectivity/results/topic_neurons --steps 990 --n 256 \
      > ../selectivity/logs/tp/neurons_$(basename $r).log 2>&1 || echo "NEURONS_FAILED $r"
  done
  echo "NEURONS_F${f}_DONE $(date)"
  T=$((30 * f)); M=""
  for a in 8 16 24 32 48; do M="$M,nk_keep_lowhack_$((T - a)),nk_keep_dnorm_$((T - a))"; done
  ls -d output/fortho_topic_gr_$b/*_s[0-9] | xargs -P 2 -I{} sh -c 'r={}; .venv/bin/python ../selectivity/eval_projection.py --run_dir $r \
    --out ../selectivity/results/topic_fsize_nk/$(basename $r).json --n_eval 256 --steps tail --modes '${M:1}' \
    > ../selectivity/logs/tp/fsize_nk_$(basename $r).log 2>&1 || echo "EVAL_FAILED $r"'
  echo "NK_F${f}_DONE $(date)"
done
echo CHAIN6_DONE
