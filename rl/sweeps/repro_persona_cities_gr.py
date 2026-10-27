"""Local reproduction of the Fig. 3 GRAFT points (pre-ablation and deployed) for
persona_qa and cities_qa: the respmon_rv2_gr_3seed runs for those two envs,
unchanged (same run names), 3 seeds each.

eval_at_start=False matches the released run_config.yaml files (train.py
defaults to True; README "Toy evaluation"). save_steps=10 (= eval_every)
saves an adapter checkpoint at every eval step, so post-hoc evals of modified
adapters can use Fig. 3's protocol (mean over the last 10% of eval steps).

Launch:
    python -u sweep.py --name repro_persona_cities_gr \
        --config sweeps/repro_persona_cities_gr.py --no_baseline --no_wandb
"""
from sweeps.respmon_rv2_gr_3seed import runs as _rv2_runs

_ENVS = ("persona_qa_flattery_respmon_bvo", "cities_qa_syco_respmon_iva")

runs = [
    {**r, "eval_at_start": False, "save_steps": 10}
    for r in _rv2_runs
    if r["run_name"].startswith(_ENVS)
]

assert len(runs) == 6, len(runs)

per_gpu = 2
no_baseline = True
