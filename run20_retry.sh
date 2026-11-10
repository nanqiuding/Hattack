#!/usr/bin/env bash
cd /root/Hattack
export HTTPS_PROXY=http://127.0.0.1:10808 HTTP_PROXY=http://127.0.0.1:10808 HF_HUB_DISABLE_XET=1
for attempt in $(seq 1 30); do
  echo "ATTEMPT $attempt $(date -Is)" >> /root/autodl-tmp/hattack20.log
  python -u gen_adv.py >> /root/autodl-tmp/hattack20.log 2>&1
  code=$?
  count=$(find cache/adv_images/DHRA -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)
  echo "EXIT $code OUTPUT_COUNT $count $(date -Is)" >> /root/autodl-tmp/hattack20.log
  if [ "$code" -eq 0 ] && [ "$count" -ge 20 ]; then exit 0; fi
  sleep 10
done
exit 1
