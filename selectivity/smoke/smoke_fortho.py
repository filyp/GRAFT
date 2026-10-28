# Smoke: persona_qa GRAFT s1, 20 steps, --forget_ortho_s 1.0; eval and checkpoint every 10.
from sweeps.repro_persona_cities_gr import runs as _runs
runs = [{**r, "max_steps": 20, "eval_every": 10, "save_steps": 10, "forget_ortho_s": 1.0,
         "run_name": "smoke_fortho_" + r["run_name"]}
        for r in _runs if r["run_name"].startswith("persona") and r["seed"] == 1]
per_gpu = 1
no_baseline = True
