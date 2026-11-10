#!/usr/bin/env bash
set -e
cd /root/Hattack
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
while [ "$(find cache/adv_images/ADVDIFFVLM_ADAPTED_NIPS100 -maxdepth 1 -name '*.png' | wc -l)" -lt 100 ]; do sleep 10; done
for model in qwen3b qwen7b llava15 llava16; do
  echo "MODEL $model $(date -Is)"
  /root/miniconda3/bin/python -u caption_open_models.py --method ADVDIFFVLM_ADAPTED_NIPS100 --model "$model" --limit 100
  /root/miniconda3/bin/python -u score_open_captions.py --method ADVDIFFVLM_ADAPTED_NIPS100 --model "$model" --limit 100
done
