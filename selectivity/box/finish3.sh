#!/bin/bash
# Unattended, per box (BOX=<name>), for sweep $SW with box-side evals (remote_evals2.sh -> selectivity/results/$OUTD):
# when the box's evals are done -> pull outputs + results -> verify every run (eval rows, tail checkpoints load,
# one forget_ortho.jsonl row per optimizer step, result JSON on the tail steps) -> tear the box down.
# The box stays up if anything fails.
SW=${SW:?}; OUTD=${OUTD:?}; N=${N:-6}; export SW
cd /srv/projects/GRAFT/selectivity/box
BOXSSH="python3 -I box.py ssh --"
RL=/srv/projects/GRAFT/rl

verify() {
  cd $RL && .venv/bin/python - "$SW" "$OUTD" "$N" <<'PY'
import glob, json, os, sys, yaml
from safetensors.torch import load_file
sw, outd, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
runs = sorted(glob.glob(f"output/{sw}/*_s[0-9]"))
assert len(runs) == n, runs
for r in runs:
    cfg = yaml.safe_load(open(f"{r}/run_config.yaml"))
    ms, ee = cfg["max_steps"], cfg["eval_every"]
    steps = sorted({json.loads(l)["step"] for l in open(f"{r}/routing_eval.jsonl") if l.strip()})
    assert steps[-1] == ms - ee and len(steps) == ms // ee - 1, (r, steps[-1], len(steps), ms)
    tail = steps[-max(1, len(steps) // 10):]
    for s in tail:
        load_file(f"{r}/checkpoint-{s}/model.safetensors")
    if "_fo0_" in os.path.basename(r):          # plain GRAFT arm: no edit, no edit log
        ortho = []
        assert not os.path.exists(f"{r}/forget_ortho.jsonl"), r
    else:
        ortho = [json.loads(l)["step"] for l in open(f"{r}/forget_ortho.jsonl")]   # '_from<f>' arms start late
        assert ortho == list(range(ortho[0], ms + 1)) and (ortho[0] == 1 or "_from" in r), (r, len(ortho), ortho[:3])
    res = json.load(open(f"../selectivity/results/{outd}/{os.path.basename(r)}.json"))
    assert res["steps"] == tail, (r, res["steps"])
    print("VERIFIED", os.path.basename(r), len(steps), "eval rows, tail", tail[0], "-", tail[-1], ",", len(ortho), "edits")
PY
}
failed() { $BOXSSH "grep -E 'FAIL|Traceback' ~/remote_run.log ~/remote_evals.log" 2>/dev/null | grep -v Warning; }

until $BOXSSH "grep -q ALL_EVALS_DONE ~/remote_evals.log" 2>/dev/null; do
  [ -n "$(failed)" ] && { echo "BOX_FAILED"; failed; exit 1; }; sleep 120
done
$BOXSSH "grep -E 'DONE|SWEEP_EXIT|EVAL_FAILED' ~/remote_run.log ~/remote_evals.log" 2>&1 | grep -v Warning
bash push.sh pull && echo "PULLED $(date)"
verify || { echo "VERIFY_FAILED: box kept up"; exit 1; }
cd /srv/projects/GRAFT/selectivity/box && python3 -I box.py teardown && echo "TORN_DOWN $(date)"
echo FINISH_DONE
