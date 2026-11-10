#!/usr/bin/env bash
cd /root/Hattack
while screen -ls | grep -Eq 'nips1000|qwen3b_pilot'; do sleep 60; done
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -u clip_geometry.py --limit 200 > /root/autodl-tmp/clip_geometry.log 2>&1
