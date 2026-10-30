#!/bin/bash
# overnight chain (reports/2026.10.29_topic_pareto.md). Run from rl/. GPU budget: neurons.py ~12 GB, an eval ~1.3 GB.
cd /srv/projects/GRAFT/rl
until grep -q ALL_DONE ../selectivity/logs/tp/topic_neurons2.log; do sleep 30; done
echo "TOPIC_NEURONS_DONE $(date)"
# persona_qa neuron analysis (6 seeds, step 990) in the background, with the persona fs/projection evals one at a time
( for s in 1 2 3 4 5 6; do r=$(ls -d output/fortho_persona_gr_[ht]/*_fo0_s$s)
    .venv/bin/python ../selectivity/neurons.py --run_dir $r --out_dir ../selectivity/results/persona_neurons --steps 990 --n 256 \
      > ../selectivity/logs/tp/pneurons_$(basename $r).log 2>&1 || echo "PNEURONS_FAILED $r"; echo "PNEURONS_DONE $r $(date)"; done ) &
PN=$!
# topic neuron-ablation evals, all modes, seeds 1,3,4,5 (s2 done separately), 2 at a time next to neurons.py
SEEDS="1 3 4 5" OUTD=topic_nk MODES=$(cat ../selectivity/topic_nk_modes.txt) P=2 bash ../selectivity/topic_nk.sh
echo "TOPIC_NK_DONE $(date)"
OUTD=persona_fs MODES=fs0.1,fs0.25,fs0.5,fs0.75,proj_upgate,qcol_upgate4,qcol_upgate16 P=1 bash ../selectivity/persona_runs.sh
echo "PERSONA_FS_DONE $(date)"
wait $PN; echo "PERSONA_NEURONS_DONE $(date)"
echo CHAIN1_DONE
