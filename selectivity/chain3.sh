#!/bin/bash
# weight-only neuron rankings (keep_dnorm, keep_wout) on the 5 topic seeds, after chain2's greedy step. Run from rl/.
cd /srv/projects/GRAFT/rl
until grep -q GREEDY_LAMS_DONE ../selectivity/logs/tp/chain2.log; do sleep 20; done
M=""; for r in keep_dnorm keep_wout; do for k in 464 456 448 440 432 416; do M="$M,nk_${r}_$k"; done; done
OUTD=topic_wonly MODES=${M:1} P=1 bash ../selectivity/topic_nk.sh
echo CHAIN3_DONE
