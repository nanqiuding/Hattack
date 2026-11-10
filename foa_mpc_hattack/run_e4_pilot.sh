#!/usr/bin/env bash
set -u
cd /root/Hattack
export HF_HOME=/root/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
while screen -ls | grep -q foa_mpc_pilot; do sleep 30; done
python_bin=/root/miniconda3/bin/python
out=/root/Hattack/cache/foa_mpc_hattack
echo "E4_SMOKE_START $(date -Is)"
"$python_bin" -u -m foa_mpc_hattack.run_attack \
  --variant E4 --num-samples 1 --steps 2 --seed 2023 --out-dir "$out/smoke"
code=$?
echo "E4_SMOKE_EXIT $code $(date -Is)"
if [ "$code" -ne 0 ]; then exit "$code"; fi
echo "E4_PILOT_START $(date -Is)"
"$python_bin" -u -m foa_mpc_hattack.run_attack \
  --variant E4 --num-samples 5 --steps 300 --seed 2023 --out-dir "$out/pilot5"
code=$?
echo "E4_PILOT_EXIT $code $(date -Is)"
exit "$code"
