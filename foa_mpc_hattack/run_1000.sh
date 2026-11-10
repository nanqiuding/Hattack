#!/usr/bin/env bash
set -euo pipefail
cd /root/Hattack
export HF_HOME=/root/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
while screen -ls | grep -Eq 'foa_mpc_100[[:space:]]'; do sleep 60; done
stage100=/root/Hattack/cache/foa_mpc_hattack/stage100
for variant in E0 E1 E2 E4; do
  for model in qwen3b qwen7b llava15 llava16; do
    test -f "$stage100/${variant}_${model}_paired_scores.csv" || {
      echo "BLOCKED: missing 100-image evaluation for $variant/$model"; exit 1;
    }
  done
done
echo "STAGE100_SUMMARY"
cat "$stage100/paired_summary.json"

out=/root/Hattack/cache/foa_mpc_hattack/stage1000
python_bin=/root/miniconda3/bin/python
for variant in E0 E1 E2 E4; do
  echo "ATTACK1000_START $variant $(date -Is)"
  "$python_bin" -u -m foa_mpc_hattack.run_attack \
    --variant "$variant" --num-samples 1000 --seed 2023 \
    --steps 300 --epsilon 16 --alpha 1 --out-dir "$out"
  echo "ATTACK1000_DONE $variant $(date -Is)"
  echo "EVAL1000_START $variant $(date -Is)"
  "$python_bin" -u -m foa_mpc_hattack.run_eval \
    --variant "$variant" --num-samples 1000 --seed 2023 --out-dir "$out"
  echo "EVAL1000_DONE $variant $(date -Is)"
done
"$python_bin" -m foa_mpc_hattack.summarize --out-dir "$out"
echo "STAGE1000_COMPLETE $(date -Is)"
