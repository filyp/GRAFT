#!/bin/bash
# persona_qa_topic S=0 runs (5 seeds): per-module edits of FORGET (reports/2026.10.29_topic_pareto.md, step 1).
# Run from rl/. Results -> selectivity/results/topic_mod/<run>.json; logs -> selectivity/logs/tp/
MODES=${MODES:-proj_up,proj_gate,proj_down,scale_upgate,perp_upgate,gsc0.5,gsc0.25,gsc0.1,qcol_up4,qcol_up16,qcol_gate4,qcol_gate16,qcol_down4,qcol_down16,qcol_upgate2,qcol_upgate4,qcol_upgate8,qcol_upgate16,proj_upgate@fs0.75,proj_upgate@fs0.5,proj_upgate@fs0.25}
OUTD=${OUTD:-topic_mod}; P=${P:-3}
ls -d output/fortho_topic_gr_t[ab]/*_fo0_s[0-9] | xargs -P $P -I{} sh -c \
  'r={}; .venv/bin/python ../selectivity/eval_projection.py --run_dir $r --out ../selectivity/results/'$OUTD'/$(basename $r).json \
     --n_eval 256 --steps tail --modes '$MODES' > ../selectivity/logs/tp/'$OUTD'_$(basename $r).log 2>&1 || echo "EVAL_FAILED $r"'
echo "ALL_DONE $(date)"
