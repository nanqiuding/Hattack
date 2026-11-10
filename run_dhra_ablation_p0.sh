#!/usr/bin/env bash
set -euo pipefail
cd /root/Hattack
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
variants=(A0 A1 A2 A3 B1 B2 B3 C1 C2)
models=(qwen3b qwen7b llava15 llava16)
for variant in "${variants[@]}"; do
 echo "STAGE20_ATTACK $variant $(date -Is)"
 /root/miniconda3/bin/python -u dhra_ablation.py --variant "$variant" --limit 20 --steps 300
 method="DHRA_ABL_${variant}"
 for model in "${models[@]}"; do
  echo "STAGE20_EVAL $variant $model $(date -Is)"
  /root/miniconda3/bin/python -u caption_open_models.py --method "$method" --model "$model" --limit 20
  /root/miniconda3/bin/python -u score_open_captions.py --method "$method" --model "$model" --limit 20
  mkdir -p "cache/ablation/stage20/${variant}/${model}"
  cp "cache/evaluation/${method}/${model}/judge_local_qwen7b/summary.json" "cache/ablation/stage20/${variant}/${model}/summary.json"
 done
done
 echo STAGE20_COMPLETE $(date -Is)
for variant in "${variants[@]}"; do
 echo "FULL_ATTACK $variant $(date -Is)"
 /root/miniconda3/bin/python -u dhra_ablation.py --variant "$variant" --limit 1000 --steps 300
 method="DHRA_ABL_${variant}"
 for model in "${models[@]}"; do
  echo "FULL_EVAL $variant $model $(date -Is)"
  /root/miniconda3/bin/python -u caption_open_models.py --method "$method" --model "$model" --limit 1000
  /root/miniconda3/bin/python -u score_open_captions.py --method "$method" --model "$model" --limit 1000
 done
done
 echo P0_COMPLETE $(date -Is)
