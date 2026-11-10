#!/usr/bin/env bash
set -euo pipefail
cd /root/Hattack
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
models=(qwen3b qwen7b llava15 llava16)
variants=(A1 A2 A3 B1 B2 B3 C1 C2)

# A0 already has 1000 completed images and judged records; summarize first 100
# without changing its 1000-image summary.
for model in "${models[@]}"; do
  /root/miniconda3/bin/python subset_ablation_summary.py \
    --method DHRA_ABL_A0 --model "$model" --limit 100 \
    --output "cache/ablation/stage100/A0/${model}/summary.json"
done

for variant in "${variants[@]}"; do
  echo "ATTACK100 $variant $(date -Is)"
  /root/miniconda3/bin/python -u dhra_ablation.py --variant "$variant" --limit 100 --steps 300
  method="DHRA_ABL_${variant}"
  for model in "${models[@]}"; do
    echo "EVAL100 $variant $model $(date -Is)"
    /root/miniconda3/bin/python -u caption_open_models.py --method "$method" --model "$model" --limit 100
    /root/miniconda3/bin/python -u score_open_captions.py --method "$method" --model "$model" --limit 100
    mkdir -p "cache/ablation/stage100/${variant}/${model}"
    cp "cache/evaluation/${method}/${model}/judge_local_qwen7b/summary.json" \
      "cache/ablation/stage100/${variant}/${model}/summary.json"
  done
done
/root/miniconda3/bin/python build_dhra_ablation_report.py --stage stage100
echo P0_100_COMPLETE $(date -Is)
