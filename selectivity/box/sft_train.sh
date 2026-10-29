#!/bin/bash
# BOX (any GPU count; DDP over all GPUs): GRAFT SFT on Terminal Wrench for $MODEL (default IFM/K2-Horizon-3.7B)
# with the paper's 32B recipe (sft/scripts/launch_gr_32b_split.sh flags) except --epochs $EPOCHS (default 2) and
# --max-length $MAX_LEN (default 8192; longer records dropped), one arm after another.
# ARMS: space-separated "<tag>:<forget_ortho_s>" (s = 0: plain GRAFT). Checkpoints: ~/ckpts/<run-name>/.
# Data: ~/sft-data (data_prep.py -> committed task_split.json -> data_prep_gr.py --recall 0.2 --clean-frac 0.5).
# Log: ~/sft_train.log. Markers: SETUP_DONE, DATA_DONE, TRAIN_DONE <run>, TRAIN_FAILED <run>, ALL_DONE.
MODEL=${MODEL:-IFM/K2-Horizon-3.7B}
EPOCHS=${EPOCHS:-2}; MAX_LEN=${MAX_LEN:-8192}
ARMS=${ARMS:-"plain:0"}
export HF_HOME=$HOME/huggingface PATH="$HOME/.local/bin:$PATH" GRAFT_SFT_DATA_DIR=$HOME/sft-data GRAFT_SFT_CKPT_DIR=$HOME/ckpts
cd ~/GRAFT/sft
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
sudo chmod 666 /var/run/docker.sock 2>/dev/null
# full eval setup (docker, submodules incl. Terminal Wrench, harbor, .venvs/eval with flash-attn): the box evaluates
# its arm after training (sft_trained_eval.sh)
bash scripts/eval/setup_box.sh > ~/setup_box.log 2>&1 || { echo "SETUP_FAILED"; tail -20 ~/setup_box.log; exit 1; }
EVAL_VENV=.venvs/eval
PY=$EVAL_VENV/bin/python
$PY -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL'); snapshot_download('Qwen/Qwen3-8B', allow_patterns=['*.json', '*.txt'])" > ~/download.log 2>&1 \
  || { echo "DOWNLOAD_FAILED"; exit 1; }
mkdir -p build
SNAP=$($PY -c "from huggingface_hub import snapshot_download; print(snapshot_download('$MODEL'))")
if grep -q "missing a thinking field" "$SNAP/chat_template.jinja" 2>/dev/null; then    # K2 Horizon
  $PY -m scripts.eval.patch_k2_template "$SNAP/chat_template.jinja" build/chat_template_patched.jinja || { echo "TEMPLATE_FAILED"; exit 1; }
  export GRAFT_CHAT_TEMPLATE=$(pwd)/build/chat_template_patched.jinja
fi
echo "SETUP_DONE $(date)"
mkdir -p $GRAFT_SFT_DATA_DIR $GRAFT_SFT_CKPT_DIR
if [[ ! -f $GRAFT_SFT_DATA_DIR/gr_train_retain.jsonl ]]; then
  ( GRAFT_TW_ROOT=third_party/terminal-wrench $PY scripts/data_prep.py
    cp data/task_split.json $GRAFT_SFT_DATA_DIR/        # the committed 232/99 split (README: do not re-split)
    PYTHONPATH=. $PY -m scripts.gr.data_prep_gr --recall 0.2 --clean-frac 0.5 ) > ~/data_prep.log 2>&1 \
    || { echo "DATA_FAILED"; tail -20 ~/data_prep.log; exit 1; }
fi
echo "DATA_DONE $(date) $(wc -l $GRAFT_SFT_DATA_DIR/gr_train_*.jsonl | tr '\n' ' ')"
source $EVAL_VENV/bin/activate
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH="$(pwd)" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NG=$(nvidia-smi -L | wc -l)
for A in $ARMS; do
  TAG=${A%%:*}; S=${A#*:}
  RUN=gr_$(basename $MODEL | tr 'A-Z.' 'a-z_')_mlp_fr02_ddp_split_ru001_ep${EPOCHS}_len${MAX_LEN}_$TAG
  EXTRA=""; [[ $S != 0 ]] && EXTRA="--forget-ortho-s $S"
  accelerate launch --config_file scripts/accelerate_ddp.yaml --num_processes $NG scripts/gr/train_gr.py \
    --model-name $MODEL --d-retain 64 --d-forget 64 --match-rslora --lora-alpha 32 --lora-r 16 --lr 5e-4 --epochs $EPOCHS \
    --warmup-ratio 0.05 --max-length $MAX_LEN --filter-overlong --step-size 48 --classifier-retain-recall 0.01 \
    --retain-from-unlabeled --unclassified-trains-both --split-moment --kappa 2 --run-name $RUN $EXTRA \
    > ~/train_$TAG.log 2>&1 && echo "TRAIN_DONE $RUN $(date)" || { echo "TRAIN_FAILED $RUN"; tail -30 ~/train_$TAG.log; }
done
echo ALL_DONE
# then this box's evals (EVALS as in sft_trained_eval.sh; "<RUN:tag>" expands to the arm's final adapter)
if [[ -n "${EVALS:-}" ]]; then
  for A in $ARMS; do TAG=${A%%:*}
    EVALS=${EVALS//"<RUN:$TAG>"/$GRAFT_SFT_CKPT_DIR/gr_$(basename $MODEL | tr 'A-Z.' 'a-z_')_mlp_fr02_ddp_split_ru001_ep${EPOCHS}_len${MAX_LEN}_$TAG/adapter_state_dict.pt}
  done
  deactivate 2>/dev/null; unset HF_HUB_OFFLINE
  EVALS="$EVALS" MODEL=$MODEL bash ~/sft_trained_eval.sh > ~/sft_trained_eval.log 2>&1
fi
