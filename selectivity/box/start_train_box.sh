#!/bin/bash
# Push the repo + sft_train.sh + sft_trained_eval.sh to box $1 and start it with ARMS="$2" (and EVALS from the env) (MODEL=${3:-IFM/K2-Horizon-3.7B}).
set -e
B=${1:?}; ARMS=${2:?}; M=${3:-IFM/K2-Horizon-3.7B}; export BOX=$B
cd /srv/projects/GRAFT/selectivity/box
bash push_sft.sh push
ST=box_state_$B.json
IP=$(python3 -c "import json;print(json.load(open('$ST'))['ip'])"); U=$(python3 -c "import json;print(json.load(open('$ST'))['user'])")
K=$(python3 -c "import json;print(json.load(open('$ST'))['key_path'])")
scp -q -i $K -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null sft_train.sh sft_trained_eval.sh $U@$IP:
python3 -I box.py ssh -- "setsid nohup env MODEL='$M' ARMS='$ARMS' EVALS='${EVALS:-}' bash ~/sft_train.sh > ~/sft_train.log 2>&1 < /dev/null &"
echo "STARTED box $B arms $ARMS model $M"
