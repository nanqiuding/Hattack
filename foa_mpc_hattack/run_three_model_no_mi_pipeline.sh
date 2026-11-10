#!/usr/bin/env bash
set -euo pipefail
cd /root/Hattack
export OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
image_dir=cache/foa_mpc_hattack/stage100/E4LDINoMIFOA_mpc_official/img/images
while screen -ls | grep -q '[.]mpc_ldinomi'; do sleep 60; done
count=$(find "$image_dir" -maxdepth 1 -name '*.png' | wc -l)
if [ "$count" -ne 100 ]; then echo "GENERATION_INCOMPLETE $count"; exit 1; fi
echo "GENERATION_COMPLETE 100 $(date -Is)"
/root/miniconda3/bin/python -m foa_mpc_hattack.run_eval \
  --variant E4LDINoMIFOA --num-samples 100 \
  --out-dir cache/foa_mpc_hattack/stage100 \
  --models qwen3b qwen7b llava15 llava16
echo "OPEN_MODEL_EVALUATION_COMPLETE $(date -Is)"
export HTTPS_PROXY=http://127.0.0.1:10808 HTTP_PROXY=http://127.0.0.1:10808
/root/miniconda3/bin/python -m foa_mpc_hattack.eval_302 \
  --root cache/foa_mpc_hattack/stage100 \
  --target-dir datas/coco_val2017_1000 \
  --output cache/evaluation/foa_mpc_hattack_302 \
  --limit 100
echo "API_EVALUATION_COMPLETE $(date -Is)"
