#!/usr/bin/env bash
set -u
cd /root/Hattack
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
while screen -ls | grep -qE '[.]mattack100[[:space:]]'; do sleep 30; done
python - <<'PY'
from pathlib import Path
from PIL import Image
root=Path('cache/adv_images/M_ATTACK_NIPS100')
assert all((root/f'{i:04d}.png').is_file() for i in range(100)), 'Missing attack output'
for i in range(100):
 with Image.open(root/f'{i:04d}.png') as im: im.verify()
print('ATTACK_IMAGES_VERIFIED 100',flush=True)
PY
if [ "$?" -ne 0 ]; then echo 'ATTACK_IMAGES_INVALID'; exit 1; fi
for model in qwen3b qwen7b llava15 llava16; do
 for stage in caption score; do
  ok=0
  for attempt in 1 2 3; do
   echo "START $model $stage attempt=$attempt $(date -Is)"
   if [ "$stage" = caption ]; then
    python -u caption_open_models.py --method M_ATTACK_NIPS100 --model "$model" --limit 100 && ok=1 && break
   else
    python -u score_open_captions.py --method M_ATTACK_NIPS100 --model "$model" --limit 100 && ok=1 && break
   fi
   echo "RETRY $model $stage $(date -Is)"
   sleep 10
  done
  if [ "$ok" -ne 1 ]; then echo "FAILED $model $stage $(date -Is)"; exit 1; fi
  echo "DONE $model $stage $(date -Is)"
 done
done
echo "ALL_COMPLETE $(date -Is)"
