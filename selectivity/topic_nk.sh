#!/bin/bash
# persona_qa_topic S=0 runs: neuron-level edits (nk_ modes need selectivity/results/topic_neurons/<run>/step*.pt;
# the nearest analysed step's ranking is used for each tail checkpoint). Run from rl/.
# Usage: MODES=... OUTD=topic_nk P=3 [SEEDS="2 3 4 5"] bash ../selectivity/topic_nk.sh
OUTD=${OUTD:?}; P=${P:-3}; SEEDS=${SEEDS:-"1 2 3 4 5"}
for s in $SEEDS; do ls -d output/fortho_topic_gr_t[ab]/*_fo0_s$s; done | xargs -P $P -I{} sh -c \
  'r={}; .venv/bin/python ../selectivity/eval_projection.py --run_dir $r --out ../selectivity/results/'$OUTD'/$(basename $r).json \
     --n_eval 256 --steps tail --modes '${MODES:?}' > ../selectivity/logs/tp/'$OUTD'_$(basename $r).log 2>&1 || echo "EVAL_FAILED $r"'
echo "ALL_DONE $(date)"
