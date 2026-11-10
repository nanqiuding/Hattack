#!/usr/bin/env bash
set -euo pipefail
cd /root/Hattack
export HTTPS_PROXY=http://127.0.0.1:10810 HTTP_PROXY=http://127.0.0.1:10810
out=cache/evaluation/foa_mpc_hattack_302_judge_gpt4o_2024_05_13
retry_run() {
  local attempt
  for attempt in $(seq 1 30); do
    if "$@"; then return 0; fi
    echo "RESTART attempt=$attempt command=$* $(date -Is)" >&2
    sleep 60
  done
  return 1
}
retry_run /root/miniconda3/bin/python -m foa_mpc_hattack.eval_302 \
  --models gpt-4o-2024-05-13 --caption-only --judge-model gpt-4o-2024-05-13 \
  --output "$out"
while screen -ls | grep -q '[.]claude_captions'; do sleep 60; done
retry_run /root/miniconda3/bin/python -m foa_mpc_hattack.eval_302 \
  --models claude-sonnet-4-6 --caption-only --judge-model gpt-4o-2024-05-13 \
  --output "$out"
while screen -ls | grep -q '[.]gemini_captions'; do sleep 60; done
/root/miniconda3/bin/python - <<'PY'
import json
from pathlib import Path
src=Path('cache/evaluation/foa_mpc_hattack_302/gemini-3.6-flash')
dst=Path('cache/evaluation/foa_mpc_hattack_302_judge_gpt4o_2024_05_13/gemini-3.6-flash')
dst.mkdir(parents=True, exist_ok=True)
for name in ('target_captions.json','MPC_noMI_LDI_captions.json',
             'FOA_MPC_LDI_captions.json','FOA_MPC_Hattack_LDI_C01_captions.json'):
    path=src/name
    count=len(json.loads(path.read_text())) if path.exists() else 0
    if count!=100: raise SystemExit(f'GEMINI_CAPTIONS_INCOMPLETE {name} {count}')
    (dst/name).write_bytes(path.read_bytes())
PY
for model in gpt-4o-2024-05-13 gemini-3.6-flash claude-sonnet-4-6; do
  retry_run /root/miniconda3/bin/python -m foa_mpc_hattack.eval_302 \
    --models "$model" --score-only --judge-model gpt-4o-2024-05-13 \
    --output "$out"
done
echo GPT4O_0513_EVALUATION_COMPLETE
