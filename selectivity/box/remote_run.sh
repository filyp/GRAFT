#!/bin/bash
# BOX: install the rl/ env (uv sync + vLLM patches), then run the 6-run sweep $SW (config sweeps/$SW.py), all runs
# concurrently. Default SW=repro_persona_cities_gr.
# Run from ~/GRAFT/rl. Log: ~/remote_run.log
set -e
SW=${SW:-repro_persona_cities_gr}
cd ~/GRAFT/rl
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync
source .venv/bin/activate
bash vllm_patches/apply.sh
python -c "import torch, vllm, flash_attn; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'vllm', vllm.__version__)"
nvidia-smi
MPS_FLAG=""
command -v nvidia-cuda-mps-control >/dev/null || MPS_FLAG="--no_mps"
echo "MPS_FLAG=$MPS_FLAG"
python -u sweep.py --name $SW --config sweeps/$SW.py \
    --no_baseline --no_wandb --per_gpu 6 $MPS_FLAG
echo SWEEP_EXIT $?
