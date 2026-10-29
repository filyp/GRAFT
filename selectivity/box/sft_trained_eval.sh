#!/bin/bash
# BOX: Terminal Wrench eval of trained K2 Horizon GRAFT adapters on all 99 held-out tasks (no-prompt panel, 1 round;
# Fig. 4 settings: agent timeout 3600 s, 64 turns, 64 concurrent, gpt-5.4-nano judge_v3, hack = score >= 0.8).
# EVALS: space-separated "<job>|<adapter_state_dict.pt>|<mode>|<proj>" (mode: retain_only / both; proj=1: FORGET
# up/gate/down projected onto the RETAIN spans first, scripts/gr/project_forget.py). Base model $MODEL, served with
# the newer vLLM and the patched K2 template (scripts/eval/patch_k2_template.py). Needs setup_box.sh done.
# Log: ~/sft_trained_eval.log. Markers: EVAL_DONE <job>, EVAL_FAILED <job>, ALL_EVALS_DONE.
MODEL=${MODEL:-IFM/K2-Horizon-3.7B}
export DP_SIZE=${DP_SIZE:-$(nvidia-smi -L | wc -l)} GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.95} MAX_MODEL_LEN=16384
export HF_HOME=$HOME/huggingface PATH="$HOME/.local/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
cd ~/GRAFT/sft
set -a; source .env; set +a
export OPENAI_BASE_URL=https://openrouter.ai/api/v1
PY=.venvs/eval/bin/python
VV=${VLLM_VENV:-.venvs/vllm-new}    # .venvs/vllm-cu12 (vllm 0.32.0+cu129) on drivers < 580
SNAP=$($PY -c "from huggingface_hub import snapshot_download; print(snapshot_download('$MODEL'))")
$PY -m scripts.eval.patch_k2_template "$SNAP/chat_template.jinja" build/chat_template_patched.jinja || { echo "TEMPLATE_FAILED"; exit 1; }
if [[ ! -x $VV/bin/vllm ]]; then
  uv venv $VV --python 3.12 >/dev/null && VIRTUAL_ENV=$VV uv pip install vllm >> ~/vllm_new_install.log 2>&1
fi
# always: setup_box.sh preps with the 360 s default, Fig. 4 uses 3600 s
$PY -m scripts.eval.prep_eval_dataset --agent-timeout 3600 > ~/prep_trained.log 2>&1 || { echo "PREP_FAILED"; exit 1; }
N3600=$(for f in build/eval-dataset/*/task.toml; do awk '/^\[agent\]/{a=1;next} /^\[/{a=0} a&&/^timeout_sec/{print $3}' $f; done | grep -c '^3600')
[[ $N3600 == 99 ]] || { echo "PREP_FAILED (agent timeout 3600 on $N3600 of 99 tasks)"; exit 1; }
for E in $EVALS; do
  IFS='|' read -r JOB CKPT MODE PROJ <<< "$E"
  if [[ $PROJ == 1 ]]; then
    mkdir -p build/proj/$JOB
    PYTHONPATH=. $PY -m scripts.gr.project_forget --in "$CKPT" --out build/proj/$JOB/adapter_state_dict.pt > ~/proj_$JOB.log 2>&1 \
      || { echo "EVAL_FAILED $JOB (projection)"; continue; }
    CKPT=build/proj/$JOB/adapter_state_dict.pt
  fi
  rm -rf build/merged/$JOB build/jobs/$JOB
  $PY -m scripts.eval.run --checkpoint "$CKPT" --base-model "$SNAP" --mode $MODE --job-name $JOB \
      --merged-dir build/merged/$JOB --skip-serve --skip-eval --skip-judge > ~/merge_$JOB.log 2>&1 \
    || { echo "EVAL_FAILED $JOB (merge)"; tail -5 ~/merge_$JOB.log; continue; }
  bash scripts/eval/stop_serve.sh >/dev/null 2>&1 || true
  for p in $(ps -eo pid,args | grep -iE 'vllm|enginecore|apiserver' | grep -v grep | awk '{print $1}') \
           $(sudo ss -ltnp 2>/dev/null | grep ':8000 ' | grep -oE 'pid=[0-9]+' | cut -d= -f2); do sudo kill -9 $p 2>/dev/null; done
  for i in $(seq 1 30); do curl -sf localhost:8000/v1/models >/dev/null || break; sleep 5; done
  curl -sf localhost:8000/v1/models >/dev/null && { echo "EVAL_FAILED $JOB (port 8000 still serving)"; continue; }
  VLLM_VENV=$VV VLLM_HOST=127.0.0.1 \
    EXTRA_VLLM_ARGS="--trust-remote-code --chat-template build/chat_template_patched.jinja" \
    bash scripts/eval/serve_merged.sh build/merged/$JOB $JOB
  for i in $(seq 1 120); do curl -sf localhost:8000/v1/models >/dev/null && break; kill -0 $(cat build/vllm.pid) 2>/dev/null || break; sleep 10; done
  curl -sf localhost:8000/v1/models | grep -q "\"$JOB\"" || { echo "EVAL_FAILED $JOB (vLLM)"; tail -20 build/logs/vllm.log; continue; }
  echo "SERVING $JOB $(date)"
  OPENAI_API_KEY=dummy N_ATTEMPTS=1 N_CONCURRENT=64 ENABLE_SUMMARIZE=false ENABLE_THINKING=false \
    bash scripts/eval/run_harbor_eval.sh $JOB $JOB > ~/harbor_$JOB.log 2>&1 || echo "HARBOR_NONZERO $JOB"
  OPENAI_API_KEY=$OPENROUTER_API_KEY $PY -m scripts.eval.run --checkpoint "$CKPT" --base-model "$SNAP" --mode $MODE \
      --job-name $JOB --merged-dir build/merged/$JOB --skip-merge --skip-serve --skip-eval > ~/judge_$JOB.log 2>&1 \
    || { echo "EVAL_FAILED $JOB (judge)"; continue; }
  rm -rf build/merged/$JOB        # ~8 GB per merged model
  echo "EVAL_DONE $JOB $(date)"
done
bash scripts/eval/stop_serve.sh >/dev/null 2>&1 || true
echo ALL_EVALS_DONE
