"""Box th of sweeps/fortho_topic_gr.py: plain GRAFT (S = 0) with a smaller FORGET adapter, mlp_config m16f4
(16 RETAIN, 4 FORGET neurons per layer instead of m16's 16/16), seeds 1-5. Run names get '_f4' after '_fo0'.
graft_w_max 5: GRAFT's kappa geometry makes the surviving adapter step at kappa_F = 20/4 = 5x lr, above the default
cap 4 (train.py refuses to start otherwise).
reports/2026.10.29_topic_pareto.md."""
import re

from sweeps.fortho_topic_gr import make

runs = [{**r, "mlp_config": "m16f4", "graft_w_max": 5.0, "run_name": re.sub(r"_fo0_s(\d+)$", r"_fo0_f4_s\1", r["run_name"])}
        for r in make((0,), seeds=(1, 2, 3, 4, 5))]
per_gpu = 6
no_baseline = True
