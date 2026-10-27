"""GRAFT persona_qa + cities_qa (3 seeds each), as sweeps/repro_persona_cities_gr.py, plus: after the optimizer
step at 25%, 50% and 75% of max_steps, the FORGET MLP-adapter up/gate weights are projected onto the RETAIN
adapter's up/gate row spans (train.py --forget_proj_at). down_proj is not touched. Run names get '_fpug'
before the seed suffix.
"""
import re

from sweeps.repro_persona_cities_gr import runs as _runs

runs = [{**r, "forget_proj_at": "0.25,0.5,0.75", "forget_proj_modules": "up,gate",
         "run_name": re.sub(r"_s(\d+)$", r"_fpug_s\1", r["run_name"])} for r in _runs]
assert len(runs) == 6 and all(r["run_name"].endswith(f"_fpug_s{r['seed']}") for r in runs)
per_gpu = 2
no_baseline = True
