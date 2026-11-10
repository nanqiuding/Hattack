#!/usr/bin/env bash
set -euo pipefail
cd /root/Hattack
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
while screen -ls | grep -q 'shared_clip_five'; do sleep 60; done
for method in SHARED_B16_NIPS100 SHARED_B16_B32_NIPS100 SHARED_PCA_NIPS100 SHARED_LEARNED_NIPS100; do
 python - "$method" <<'PY'
from pathlib import Path
from PIL import Image
import sys
root=Path('cache/adv_images')/sys.argv[1]
for i in range(100):
 path=root/f'{i:04d}.png'
 assert path.is_file(),path
 with Image.open(path) as image: image.verify()
print('VERIFIED',sys.argv[1],flush=True)
PY
 for model in qwen3b qwen7b llava15 llava16; do
  echo "EVAL_START $method $model $(date -Is)"
  python -u caption_open_models.py --method "$method" --model "$model" --limit 100
  python -u score_open_captions.py --method "$method" --model "$model" --limit 100
  echo "EVAL_DONE $method $model $(date -Is)"
 done
done
echo "ALL_COMPLETE $(date -Is)"
