#!/usr/bin/env bash
set -uo pipefail
cd /root/Hattack
export HF_HOME=/root/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python_bin=/root/miniconda3/bin/python
out=/root/Hattack/cache/foa_mpc_hattack

"$python_bin" -m unittest discover -s foa_mpc_hattack/tests -v
for model in CLIP_B16 CLIP_B32 CLIP_LAION DINOv2 InternVL3_1B; do
  echo "FEATURE_CHECK_START $model $(date -Is)"
  "$python_bin" -u -m foa_mpc_hattack.check_features --model "$model"
  echo "FEATURE_CHECK_EXIT $model $? $(date -Is)"
done
declare -A smoke_ok
for variant in E0 E1 E2 E4; do
  echo "SMOKE_START $variant $(date -Is)"
  "$python_bin" -u -m foa_mpc_hattack.run_attack \
    --variant "$variant" --num-samples 1 --steps 2 \
    --seed 2023 --out-dir "$out/smoke"
  code=$?
  smoke_ok[$variant]=$code
  echo "SMOKE_EXIT $variant $code $(date -Is)"
done

for variant in E0 E1 E2 E4; do
  if [ "${smoke_ok[$variant]}" -ne 0 ]; then
    echo "PILOT_SKIPPED $variant smoke_failed"
    continue
  fi
  echo "PILOT_START $variant $(date -Is)"
  "$python_bin" -u -m foa_mpc_hattack.run_attack \
    --variant "$variant" --num-samples 5 --steps 300 \
    --seed 2023 --out-dir "$out/pilot5"
  code=$?
  echo "PILOT_EXIT $variant $code $(date -Is)"
done
