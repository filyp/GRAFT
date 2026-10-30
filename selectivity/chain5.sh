#!/bin/bash
# validation set for choosing k: train-split prompts, seed 11 (the neuron rankings used train seed 7), 256 prompts,
# same tail checkpoints; forget-scale curve + neuron ablations on the 5 topic seeds. After chains 2 and 3. From rl/.
cd /srv/projects/GRAFT/rl
until grep -q CHAIN2_DONE ../selectivity/logs/tp/chain2.log && grep -q CHAIN3_DONE ../selectivity/logs/tp/chain3.log; do sleep 30; done
M="both,retain_only,fs0.1,fs0.25,fs0.5,fs0.75,proj_upgate,qcol_upgate16"
for k in 464 456 448 440 432 424 416; do M="$M,nk_keep_lowhack_$k"; done
for k in 464 456 448 440 432 424; do M="$M,nk_keep_greedy0_$k,nk_keep_greedy0.05_$k"; done
QA_EVAL_SPLIT=train QA_EVAL_SEED=11 OUTD=topic_val MODES=$M P=2 bash ../selectivity/topic_nk.sh
echo CHAIN5_DONE
