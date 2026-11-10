#!/usr/bin/env bash
set -euo pipefail
cd /root/Hattack
export HTTPS_PROXY=http://127.0.0.1:10808 HTTP_PROXY=http://127.0.0.1:10808 HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=120 HF_HUB_ETAG_TIMEOUT=30
python - <<'PY'
import time
from pathlib import Path
from huggingface_hub import hf_hub_download
target=Path('/root/autodl-tmp/sd_v1_4/unet/diffusion_pytorch_model.safetensors')
expected=3438167534
for attempt in range(20):
 if target.exists() and target.stat().st_size==expected: break
 try:
  hf_hub_download('CompVis/stable-diffusion-v1-4','unet/diffusion_pytorch_model.safetensors',local_dir='/root/autodl-tmp/sd_v1_4')
 except Exception as e: print('DOWNLOAD_RETRY',attempt,repr(e),flush=True)
 print('CURRENT_SIZE',target.stat().st_size if target.exists() else 0,flush=True)
 time.sleep(10)
assert target.exists() and target.stat().st_size==expected,'UNET_INCOMPLETE'
print('UNET_VERIFIED',expected,flush=True)
PY
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -u diffusion_inverse_pilot.py
