#!/bin/bash
# greedy lam 0.05 / 0.2 / 1 (sign fixed 22:35), then generation evals of the greedy rankings (lam 0, 0.05, 0.2, 1)
# on the 5 topic seeds. Run from rl/.
cd /srv/projects/GRAFT/rl
for lam in 0.05 0.2 1; do LAM=$lam bash ../selectivity/topic_greedy.sh > ../selectivity/logs/tp/topic_greedy$lam.log 2>&1; done
echo "GREEDY_LAMS_DONE $(date)"
M=""
for lam in 0 0.05 0.2 1; do for k in 472 464 456 448 440 432 424 416 400 384; do M="$M,nk_keep_greedy${lam}_$k"; done; done
OUTD=topic_greedy MODES=${M:1} P=2 bash ../selectivity/topic_nk.sh
echo CHAIN2_DONE
