#!/usr/bin/env bash
set -euo pipefail
cd /root/Hattack
export HF_HOME=/root/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
while screen -ls | grep -Eq 'foa_mpc_eval[[:space:]]'; do sleep 30; done
python_bin=/root/miniconda3/bin/python
out=/root/Hattack/cache/foa_mpc_hattack
echo "E4_SMOKE_RETRY_START $(date -Is)"
"$python_bin" -u -m foa_mpc_hattack.run_attack \
  --variant E4 --num-samples 1 --steps 2 --seed 2023 --out-dir "$out/smoke"
echo "E4_SMOKE_RETRY_DONE $(date -Is)"
echo "E4_PILOT_RETRY_START $(date -Is)"
"$python_bin" -u -m foa_mpc_hattack.run_attack \
  --variant E4 --num-samples 5 --steps 300 --seed 2023 --out-dir "$out/pilot5"
echo "E4_PILOT_RETRY_DONE $(date -Is)"
"$python_bin" -u -m foa_mpc_hattack.run_eval \
  --variant E4 --num-samples 5 --seed 2023 --out-dir "$out/pilot5"
"$python_bin" -m foa_mpc_hattack.summarize --out-dir "$out/pilot5"
echo "E4_RETRY_COMPLETE $(date -Is)"
