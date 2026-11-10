#!/usr/bin/env bash
cd /root/Hattack
export HTTPS_PROXY=http://127.0.0.1:10808 HTTP_PROXY=http://127.0.0.1:10808 HF_HUB_DISABLE_XET=1
python -u prepare_200.py > /root/autodl-tmp/hattack200.log 2>&1 || exit 1
sed -i 's/num_samples = 20/num_samples = 200/' config.toml
for attempt in $(seq 1 5); do
 echo "ATTEMPT $attempt $(date -Is)" >> /root/autodl-tmp/hattack200.log
 python -u gen_adv.py >> /root/autodl-tmp/hattack200.log 2>&1
 code=$?
 count=$(find cache/adv_images/DHRA -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)
 echo "EXIT $code OUTPUT_COUNT $count $(date -Is)" >> /root/autodl-tmp/hattack200.log
 if [ "$code" -eq 0 ] && [ "$count" -ge 200 ]; then exit 0; fi
 sleep 10
done
exit 1
