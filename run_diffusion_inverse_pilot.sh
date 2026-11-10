#!/usr/bin/env bash
set -euo pipefail
cd /root/Hattack
while screen -ls | grep -qE '[.]sd_download[[:space:]]'; do sleep 30; done
test -f /root/autodl-tmp/sd_v1_4/unet/diffusion_pytorch_model.safetensors
test -f /root/autodl-tmp/sd_v1_4/vae/diffusion_pytorch_model.safetensors
test -f /root/autodl-tmp/sd_v1_4/text_encoder/model.safetensors
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -u diffusion_inverse_pilot.py
