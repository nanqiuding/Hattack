#!/usr/bin/env bash
set -euo pipefail
cd /root/Hattack
export HTTPS_PROXY=http://127.0.0.1:10810 HTTP_PROXY=http://127.0.0.1:10810
out=cache/evaluation/foa_mpc_hattack_302
while screen -ls | grep -q '[.]gemini_captions'; do sleep 60; done
/root/miniconda3/bin/python - <<'PY'
import json
from pathlib import Path
p=Path('cache/evaluation/foa_mpc_hattack_302/gemini-3.6-flash')
for name in ('target_captions.json','MPC_noMI_LDI_captions.json',
             'FOA_MPC_LDI_captions.json','FOA_MPC_Hattack_LDI_C01_captions.json'):
    path=p/name
    count=len(json.loads(path.read_text())) if path.exists() else 0
    if count!=100: raise SystemExit(f'GEMINI_CAPTIONS_INCOMPLETE {name} {count}')
PY
/root/miniconda3/bin/python -m foa_mpc_hattack.eval_302 \
  --models gemini-3.6-flash --score-only --judge-model gpt-4o-mini \
  --output "$out"
/root/miniconda3/bin/python -m foa_mpc_hattack.eval_302 \
  --models gpt-4o-mini --caption-only --judge-model gpt-4o-mini \
  --output "$out"
/root/miniconda3/bin/python -m foa_mpc_hattack.eval_302 \
  --models gpt-4o-mini --score-only --judge-model gpt-4o-mini \
  --output "$out"
echo MINI_EVALUATION_COMPLETE
