#!/bin/bash
# original persona_qa transfer (6 seeds): after chain1 (neurons + fs/proj evals) and chain2's greedy step,
# greedy lam 0 / 0.2 on persona, then neuron-ablation evals. Run from rl/.
cd /srv/projects/GRAFT/rl
until grep -q CHAIN1_DONE ../selectivity/logs/tp/chain1.log && grep -q GREEDY_LAMS_DONE ../selectivity/logs/tp/chain2.log; do sleep 30; done
for lam in 0 0.2; do
  SEEDS="1 2 3 4 5 6" RUNGLOB="output/fortho_persona_gr_[ht]" ND=persona_neurons LAM=$lam bash ../selectivity/topic_greedy.sh \
    > ../selectivity/logs/tp/persona_greedy$lam.log 2>&1
done
echo "PERSONA_GREEDY_DONE $(date)"
M="nk_keep_lowhack_464,nk_keep_lowhack_456,nk_keep_lowhack_448,nk_keep_lowhack_440,nk_keep_lowhack_432,nk_keep_lowhack_416,nk_keep_lowhack_400"
M="$M,nk_keep_dnorm_456,nk_keep_dnorm_448,nk_keep_dnorm_432,nk_keep_wout_448,nk_keep_wout_432,nk_random_448,nk_random1_448,nk_random2_448"
for lam in 0 0.2; do for k in 464 456 448 440 432 416; do M="$M,nk_keep_greedy${lam}_$k"; done; done
OUTD=persona_nk MODES=$M P=2 bash ../selectivity/persona_runs.sh
echo CHAIN4_DONE
