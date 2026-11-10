#!/usr/bin/env bash
set -euo pipefail
cd /root/Hattack
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
while screen -ls | grep -q 'mattack100'; do sleep 30; done
python - <<'PY'
from pathlib import Path
root=Path('cache/adv_images/M_ATTACK_NIPS100')
assert all((root/f'{i:04d}.png').is_file() for i in range(100))
print('M_ATTACK_BASELINE_READY 100',flush=True)
PY
echo "FIT_START $(date -Is)"
python -u shared_clip_experiment.py fit
echo "FIT_DONE $(date -Is)"
for variant in single_b16 aligned_b16_b32 aligned_pca aligned_learned; do
 echo "ATTACK_START $variant $(date -Is)"
 python -u shared_clip_experiment.py attack --variant "$variant"
 echo "ATTACK_DONE $variant $(date -Is)"
done
echo "ALL_COMPLETE $(date -Is)"
