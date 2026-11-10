#!/usr/bin/env bash
set -u
cd /root/Hattack
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
for model in qwen3b qwen7b llava15 llava16; do
  for stage in caption score; do
    ok=0
    for attempt in 1 2 3; do
      echo "START $model $stage attempt=$attempt $(date -Is)"
      if [ "$stage" = caption ]; then
        python -u caption_open_models.py --model "$model" --limit 1000 && ok=1 && break
      else
        python -u score_open_captions.py --model "$model" --limit 1000 && ok=1 && break
      fi
      echo "RETRY $model $stage $(date -Is)"
      sleep 10
    done
    if [ "$ok" -ne 1 ]; then echo "FAILED $model $stage $(date -Is)"; exit 1; fi
    echo "DONE $model $stage $(date -Is)"
  done
done
echo "ALL_COMPLETE $(date -Is)"
