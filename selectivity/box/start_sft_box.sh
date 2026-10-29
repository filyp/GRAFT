#!/bin/bash
# Per box: push the repo + sft_base_eval.sh to box $1 and start it for model $2 with DP=$3 (default 1).
# Usage: bash start_sft_box.sh <box> <hf model id> [dp]
set -e
B=${1:?}; M=${2:?}; DP=${3:-1}; export BOX=$B
cd /srv/projects/GRAFT/selectivity/box
bash push_sft.sh push
ST=box_state_$B.json
IP=$(python3 -c "import json;print(json.load(open('$ST'))['ip'])"); U=$(python3 -c "import json;print(json.load(open('$ST'))['user'])")
K=$(python3 -c "import json;print(json.load(open('$ST'))['key_path'])")
scp -q -i $K -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null sft_base_eval.sh $U@$IP:
python3 -I box.py ssh -- "setsid nohup env MODELS='$M' DP_SIZE=$DP bash ~/sft_base_eval.sh > ~/sft_base_eval.log 2>&1 < /dev/null &"
echo "STARTED box $B model $M dp $DP"
