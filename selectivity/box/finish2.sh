#!/bin/bash
# Unattended, for sweep $SW with box-side evals (remote_evals.sh -> selectivity/results/$OUTD):
# as each env's evals finish on the box -> pull outputs + results -> verify checkpoints and results.
# When all 6 runs verify: tear the box down. The box stays up if anything fails.
SW=${SW:?}; OUTD=${OUTD:?}; export SW
cd /srv/projects/GRAFT/selectivity/box
BOX="python3 -I box.py ssh --"
RL=/srv/projects/GRAFT/rl

verify() {   # $1 = env prefix
  cd $RL && .venv/bin/python - "$1" "$SW" "$OUTD" <<'PY'
import glob, json, os, sys, yaml
from safetensors.torch import load_file
env, sw, outd = sys.argv[1:]
runs = sorted(glob.glob(f"output/{sw}/{env}*_s[0-9]"))
assert len(runs) == 3, runs
for r in runs:
    cfg = yaml.safe_load(open(f"{r}/run_config.yaml"))
    ms, ee = cfg["max_steps"], cfg["eval_every"]
    steps = sorted({json.loads(l)["step"] for l in open(f"{r}/routing_eval.jsonl") if l.strip()})
    assert steps[-1] == ms - ee and len(steps) == ms // ee - 1, (r, steps[-1], len(steps), ms)
    tail = steps[-max(1, len(steps) // 10):]
    for s in tail:
        load_file(f"{r}/checkpoint-{s}/model.safetensors")
    proj = [json.loads(l)["step"] for l in open(f"{r}/forget_proj.jsonl")]
    assert proj == [ms // 4, ms // 2, 3 * ms // 4], (r, proj)
    res = json.load(open(f"../selectivity/results/{outd}/{os.path.basename(r)}.json"))
    assert res["steps"] == tail, (r, res["steps"])
    print("VERIFIED", os.path.basename(r), len(steps), "eval rows, tail", tail[0], "-", tail[-1], "proj at", proj)
PY
}
failed() { $BOX "grep -E 'FAIL|Traceback' ~/remote_run.log ~/remote_evals.log" 2>/dev/null | grep -v Warning; }

until $BOX "grep -q 'EVALS_DONE persona_qa' ~/remote_evals.log" 2>/dev/null; do
  [ -n "$(failed)" ] && { echo "BOX_FAILED"; failed; exit 1; }; sleep 120
done
bash push.sh pull && echo "PULLED_A $(date)"
verify persona_qa || { echo "VERIFY_FAILED persona"; exit 1; }

cd /srv/projects/GRAFT/selectivity/box
until $BOX "grep -q ALL_EVALS_DONE ~/remote_evals.log" 2>/dev/null; do
  [ -n "$(failed)" ] && { echo "BOX_FAILED"; failed; exit 1; }; sleep 120
done
$BOX "grep -E 'DONE|SWEEP_EXIT' ~/remote_run.log ~/remote_evals.log" 2>&1 | grep -v Warning
bash push.sh pull && echo "PULLED_B $(date)"
verify persona_qa && verify cities_qa || { echo "VERIFY_FAILED: box kept up"; exit 1; }
cd /srv/projects/GRAFT/selectivity/box && python3 -I box.py teardown && echo "TORN_DOWN $(date)"
echo FINISH_DONE
