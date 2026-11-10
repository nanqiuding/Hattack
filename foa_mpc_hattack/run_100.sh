#!/usr/bin/env bash
set -euo pipefail
cd /root/Hattack
export HF_HOME=/root/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
while screen -ls | grep -Eq 'foa_mpc_eval[[:space:]]|foa_mpc_e4_retry[[:space:]]'; do sleep 30; done
pilot=/root/Hattack/cache/foa_mpc_hattack/pilot5
for variant in E0 E1 E2 E4; do
  for model in qwen3b qwen7b llava15 llava16; do
    test -f "$pilot/${variant}_${model}_paired_scores.csv" || {
      echo "BLOCKED: missing 5-image evaluation for $variant/$model"; exit 1;
    }
  done
done

out=/root/Hattack/cache/foa_mpc_hattack/stage100
python_bin=/root/miniconda3/bin/python
for variant in E0 E1 E2 E4; do
  echo "ATTACK100_START $variant $(date -Is)"
  "$python_bin" -u -m foa_mpc_hattack.run_attack \
    --variant "$variant" --num-samples 100 --seed 2023 \
    --steps 300 --epsilon 16 --alpha 1 --out-dir "$out"
  echo "ATTACK100_DONE $variant $(date -Is)"
  echo "EVAL100_START $variant $(date -Is)"
  "$python_bin" -u -m foa_mpc_hattack.run_eval \
    --variant "$variant" --num-samples 100 --seed 2023 --out-dir "$out"
  echo "EVAL100_DONE $variant $(date -Is)"
done
"$python_bin" -m foa_mpc_hattack.summarize --out-dir "$out"
echo "STAGE100_COMPLETE $(date -Is)"
