#!/bin/bash
# Unattended: as each env's 3 box runs finish -> pull -> verify the checkpoints Fig. 3's protocol needs ->
# local post-hoc evals (tail checkpoints, n=256, all modes; skipped if the result exists). When the whole
# sweep is done and all 6 runs verify: tear the box down, then plot. The box stays up if anything fails.
cd /srv/projects/GRAFT/selectivity/box
BOX="python3 -I box.py ssh --"
RL=/srv/projects/GRAFT/rl
RES=/srv/projects/GRAFT/selectivity/results/posthoc_tail

verify() {   # $1 = glob of run dirs under output/repro_persona_cities_gr
  cd $RL && .venv/bin/python - "$1" <<'EOF'
import glob, json, sys, yaml
from safetensors.torch import load_file
runs = sorted(glob.glob(f"output/repro_persona_cities_gr/{sys.argv[1]}"))
assert len(runs) == 3, runs
for r in runs:
    cfg = yaml.safe_load(open(f"{r}/run_config.yaml"))
    ms, ee = cfg["max_steps"], cfg["eval_every"]
    steps = sorted({json.loads(l)["step"] for l in open(f"{r}/routing_eval.jsonl") if l.strip()})
    # the last in-training eval row is at max_steps - eval_every (as in the paper's released runs)
    assert steps[-1] == ms - ee and len(steps) == ms // ee - 1, (r, steps[-1], len(steps), ms)
    tail = steps[-max(1, len(steps) // 10):]
    for s in tail:
        load_file(f"{r}/checkpoint-{s}/model.safetensors")
    n_ck = len(glob.glob(f"{r}/checkpoint-*/model.safetensors"))
    print("VERIFIED", r.split("/")[-1], len(steps), "eval rows,", n_ck, "checkpoints, tail", tail[0], "-", tail[-1])
EOF
}

evals() {    # $1 = run-name prefix
  cd $RL && source .venv/bin/activate
  for r in output/repro_persona_cities_gr/$1*_s[0-9]; do
    [ -f $RES/$(basename $r).json ] && continue
    python ../selectivity/eval_projection.py --run_dir $r --out $RES/$(basename $r).json --n_eval 256 --steps tail \
        > /srv/projects/GRAFT/selectivity/logs/eval_$(basename $r).log 2>&1 || echo "EVAL_FAILED $r"
    echo "EVALUATED $(basename $r) $(date)"
  done
}

done_count() { $BOX "grep -c 'DONE .*$1.* OK' ~/remote_run.log" 2>/dev/null | tail -1; }

# phase A: persona_qa (1000 steps) finishes first
until [ "$(done_count persona_qa)" = "3" ]; do
  $BOX "grep -q 'FAIL(exit' ~/remote_run.log" 2>/dev/null && { echo "RUN_FAILED on box"; $BOX "grep 'FAIL(exit' ~/remote_run.log"; exit 1; }
  sleep 120
done
bash push.sh pull && echo "PULLED_A $(date)"
verify "persona_qa*_s[0-9]" || { echo "VERIFY_FAILED persona"; exit 1; }
evals persona_qa

# phase B: whole sweep
cd /srv/projects/GRAFT/selectivity/box
until $BOX "grep -q SWEEP_EXIT ~/remote_run.log" 2>/dev/null; do sleep 120; done
$BOX "grep -E 'DONE|SWEEP_EXIT' ~/remote_run.log" 2>&1 | grep -v Warning
bash push.sh pull && echo "PULLED_B $(date)"
verify "persona_qa*_s[0-9]" && verify "cities_qa*_s[0-9]" || { echo "VERIFY_FAILED: box kept up"; exit 1; }
cd /srv/projects/GRAFT/selectivity/box && python3 -I box.py teardown && echo "TORN_DOWN $(date)"
evals cities_qa
echo FINISH_DONE
