#!/bin/bash
# BOX: base-model Terminal Wrench eval (a fixed random N_TASKS=25 of the 99 held-out tasks, seed 0; no-prompt panel, 1 round, Fig. 4 settings: agent timeout
# 3600 s, 64 turns, 64 concurrent, non-thinking, gpt-5.4-nano judge_v3, hack = score >= 0.8) for each model in
# $MODELS (default: Qwen/Qwen3-8B IFM/K2-Horizon-3.7B IFM/K2-Horizon-7B Qwen/Qwen3.5-9B), one after another, DP=$DP_SIZE.
# Qwen3-* runs on the pinned vLLM 0.9.2; other models (K2 Horizon, Qwen3.5) on a newer vLLM (.venvs/vllm-new, --trust-remote-code).
# Log: ~/sft_base_eval.log. Markers: SETUP_DONE, EVAL_DONE <job>, EVAL_FAILED <job>, ALL_DONE.
MODELS=${MODELS:-"Qwen/Qwen3-8B IFM/K2-Horizon-3.7B IFM/K2-Horizon-7B Qwen/Qwen3.5-9B"}
export DP_SIZE=${DP_SIZE:-2} GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.95} MAX_MODEL_LEN=16384
export HF_HOME=$HOME/huggingface PATH="$HOME/.local/bin:$PATH"
cd ~/GRAFT/sft
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
command -v docker >/dev/null || { sudo apt-get update -y && sudo apt-get install -y docker.io docker-compose-v2; }
sudo chmod 666 /var/run/docker.sock          # ephemeral box: instead of a docker-group re-login
bash scripts/eval/setup_box.sh > ~/setup_box.log 2>&1 || { echo "SETUP_FAILED"; tail -20 ~/setup_box.log; exit 1; }
git -C third_party/harbor apply --check -R ../../scripts/eval/patches/harbor_verifier_on_ctx_overflow.patch \
  || { echo "PATCH_NOT_APPLIED"; exit 1; }      # reverse-applies cleanly iff the patch is in
.venvs/eval/bin/python -m scripts.eval.prep_eval_dataset --agent-timeout 3600 > ~/prep.log 2>&1 || { echo "PREP_FAILED"; exit 1; }
# keep a fixed random subset of N_TASKS (default 25) of the 99 eval tasks (seed 0), the same on every box
N_TASKS=${N_TASKS:-25} .venvs/eval/bin/python - <<'PY' || { echo "SUBSET_FAILED"; exit 1; }
import os, random, shutil
d = "build/eval-dataset"; tasks = sorted(t for t in os.listdir(d) if os.path.isdir(os.path.join(d, t)))
assert len(tasks) == 99, len(tasks)
keep = set(random.Random(0).sample(tasks, int(os.environ["N_TASKS"])))
for t in tasks:
    if t not in keep: shutil.rmtree(os.path.join(d, t))
open(os.path.expanduser("~/subset_tasks.txt"), "w").write("\n".join(sorted(keep)) + "\n")
print("SUBSET", len(keep), "tasks")
PY
echo "SETUP_DONE $(date)"
set -a; source .env; set +a
export OPENAI_BASE_URL=https://openrouter.ai/api/v1
for M in $MODELS; do
  NAME=base-$(basename $M | tr 'A-Z' 'a-z'); JOB=$NAME-no; [[ ${N_TASKS:-25} != 25 ]] && JOB=$JOB-n$N_TASKS
  SNAP=$(.venvs/eval/bin/python -c "from huggingface_hub import snapshot_download; print(snapshot_download('$M'))" 2>>~/download.log)
  # stop_serve.sh kills only the parent; the DP API servers (VLLM::APIServer_*) keep port 8000
  bash scripts/eval/stop_serve.sh >/dev/null 2>&1 || true
  for p in $(sudo ss -ltnp 2>/dev/null | grep ':8000 ' | grep -oE 'pid=[0-9]+' | cut -d= -f2); do sudo kill -9 $p 2>/dev/null; done
  for i in $(seq 1 30); do curl -sf localhost:8000/v1/models >/dev/null || break; sleep 5; done   # old server gone
  if curl -sf localhost:8000/v1/models >/dev/null; then echo "EVAL_FAILED $JOB (port 8000 still serving)"; continue; fi
  if [[ $M == Qwen/Qwen3-* ]]; then
    bash scripts/eval/serve_merged.sh "$SNAP" "$NAME"
  else
    if [[ ! -x .venvs/vllm-new/bin/vllm ]]; then
      uv venv .venvs/vllm-new --python 3.12 >/dev/null && VIRTUAL_ENV=.venvs/vllm-new uv pip install vllm >> ~/vllm_new_install.log 2>&1
    fi
    # K2 Horizon's chat template raises on assistant turns without a thinking field (terminus-2 replays its earlier
    # turns without one): scripts/eval/patch_k2_template.py renders them with an empty think_faster block and
    # defaults reasoning_effort to 'low' (same template as K2 training)
    CT_ARG=""
    if [[ -f "$SNAP/chat_template.jinja" ]] && grep -q "missing a thinking field" "$SNAP/chat_template.jinja"; then
      .venvs/eval/bin/python -m scripts.eval.patch_k2_template "$SNAP/chat_template.jinja" build/chat_template_patched.jinja \
        || { echo "EVAL_FAILED $JOB (template patch)"; continue; }
      CT_ARG="--chat-template build/chat_template_patched.jinja"
    fi
    # FlashInfer's sampler JIT-compiles with the system nvcc, which can be older than the CUDA 13 build (verda:
    # "nvcc fatal: Unknown option '--compress-mode=size'"); vLLM's own sampler needs no compile
    VLLM_USE_FLASHINFER_SAMPLER=0 nohup .venvs/vllm-new/bin/vllm serve "$SNAP" --served-model-name "$NAME" --trust-remote-code --host 127.0.0.1 \
      --port 8000 --max-model-len $MAX_MODEL_LEN --gpu-memory-utilization $GPU_MEM_UTIL --data-parallel-size $DP_SIZE $CT_ARG \
      > build/logs/vllm.log 2>&1 &
    echo $! > build/vllm.pid
  fi
  for i in $(seq 1 120); do curl -sf localhost:8000/v1/models >/dev/null && break; kill -0 $(cat build/vllm.pid) 2>/dev/null || break; sleep 10; done
  kill -0 $(cat build/vllm.pid) 2>/dev/null || { echo "EVAL_FAILED $JOB (vLLM exited)"; tail -30 build/logs/vllm.log; continue; }
  if ! curl -sf localhost:8000/v1/models >/dev/null; then echo "EVAL_FAILED $JOB (vLLM did not come up)"; tail -30 build/logs/vllm.log; continue; fi
  echo "SERVING $NAME $(date)"
  OPENAI_API_KEY=dummy N_ATTEMPTS=1 N_CONCURRENT=64 ENABLE_SUMMARIZE=false ENABLE_THINKING=false \
    bash scripts/eval/run_harbor_eval.sh "$NAME" "$JOB" > ~/harbor_$JOB.log 2>&1 || echo "HARBOR_NONZERO $JOB"
  OPENAI_API_KEY=$OPENROUTER_API_KEY .venvs/eval/bin/python -m scripts.eval.score_rollouts build/jobs/$JOB \
    --judge-model openai/gpt-5.4-nano --prompt judge_v3 --output build/jobs/$JOB/judge_scores_judge_v3.json \
    > ~/judge_$JOB.log 2>&1 || { echo "EVAL_FAILED $JOB (judge)"; continue; }
  .venvs/eval/bin/python -m scripts.eval.recompute_summary build/jobs/$JOB --scores-file judge_scores_judge_v3.json \
    > ~/summary_$JOB.log 2>&1
  cp ~/subset_tasks.txt build/jobs/$JOB/ 2>/dev/null
  echo "EVAL_DONE $JOB $(date)"
done
bash scripts/eval/stop_serve.sh >/dev/null 2>&1 || true
for p in $(sudo ss -ltnp 2>/dev/null | grep ':8000 ' | grep -oE 'pid=[0-9]+' | cut -d= -f2); do sudo kill -9 $p 2>/dev/null; done
echo ALL_DONE
