#!/usr/bin/env bash
cd /root/Hattack
while screen -ls | grep -q nips1000; do sleep 60; done
if ! grep -q COMPLETE /root/autodl-tmp/qwen3b_download.log; then
 echo "Qwen model download incomplete" > /root/autodl-tmp/qwen3b_pilot.log
 exit 1
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -u caption_open_model.py --method DHRA --target-dir datas/target_1000 --limit 20 > /root/autodl-tmp/qwen3b_pilot.log 2>&1
