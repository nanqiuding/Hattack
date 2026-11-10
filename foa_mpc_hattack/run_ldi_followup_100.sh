#!/usr/bin/env bash
set -euo pipefail
cd /root/Hattack
export HF_HOME=/root/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python_bin=/root/miniconda3/bin/python
out=/root/Hattack/cache/foa_mpc_hattack/stage100

while screen -ls | grep -Eq 'foa_ldi_100[[:space:]]'; do sleep 30; done
for model in qwen3b qwen7b llava15 llava16; do
  test -f "$out/E1LDI_${model}_paired_scores.csv" || {
    echo "BLOCKED: E1LDI evaluation missing for $model"
    exit 1
  }
done

"$python_bin" -u -m foa_mpc_hattack.run_attack \
  --variant E4LDI --num-samples 1 --steps 2 --seed 2023 \
  --out-dir /root/Hattack/cache/foa_mpc_hattack/ldi_followup_smoke
"$python_bin" -u -m foa_mpc_hattack.run_attack \
  --variant E3LDI_C01 --num-samples 1 --steps 2 --seed 2023 \
  --out-dir /root/Hattack/cache/foa_mpc_hattack/ldi_followup_smoke

for variant in E4LDI E3LDI_C01 E3LDI_C001 E3LDI_C1 E3LDI_C0001; do
  echo "ATTACK_START $variant $(date -Is)"
  "$python_bin" -u -m foa_mpc_hattack.run_attack \
    --variant "$variant" --num-samples 100 --steps 300 \
    --seed 2023 --epsilon 16 --alpha 1 --out-dir "$out"
  echo "ATTACK_DONE $variant $(date -Is)"
  "$python_bin" -u -m foa_mpc_hattack.run_eval \
    --variant "$variant" --num-samples 100 --seed 2023 --out-dir "$out"
  echo "EVAL_DONE $variant $(date -Is)"
done
"$python_bin" -m foa_mpc_hattack.summarize --out-dir "$out"
echo "LDI_FOLLOWUP_COMPLETE $(date -Is)"
