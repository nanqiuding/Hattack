#!/usr/bin/env bash
set -euo pipefail
cd /root/Hattack
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
while screen -ls | grep -qE '[.]mattack1000[[:space:]]'; do sleep 60; done
python - <<'PY'
from pathlib import Path
from PIL import Image
import json
src=Path('cache/adv_images/M_ATTACK_NIPS1000')
assert all((src/f'{i:04d}.png').is_file() for i in range(1000)), 'Missing attack output'
for i in range(1000):
 with Image.open(src/f'{i:04d}.png') as im: im.verify()
print('ATTACK_IMAGES_VERIFIED 1000',flush=True)
for model in ('qwen3b','qwen7b','llava15','llava16'):
 old=Path('cache/evaluation/M_ATTACK_NIPS100')/model/'judge_local_qwen7b'/'records.json'
 new=Path('cache/evaluation/M_ATTACK_NIPS1000')/model/'judge_local_qwen7b'/'records.json'
 if new.exists(): continue
 data=json.loads(old.read_text())
 for item in data.values():
  item['adversarial']=item['adversarial'].replace('M_ATTACK_NIPS100/','M_ATTACK_NIPS1000/')
 new.parent.mkdir(parents=True,exist_ok=True)
 new.write_text(json.dumps(data,ensure_ascii=False,indent=2))
 print('REUSED',model,len(data),flush=True)
PY
for model in qwen3b qwen7b llava15 llava16; do
 echo "EVAL_START $model $(date -Is)"
 python -u caption_open_models.py --method M_ATTACK_NIPS1000 --model "$model" --limit 1000
 python -u score_open_captions.py --method M_ATTACK_NIPS1000 --model "$model" --limit 1000
 echo "EVAL_DONE $model $(date -Is)"
done
echo "ALL_COMPLETE $(date -Is)"
