#!/usr/bin/env bash
cd /root/Hattack
export HTTPS_PROXY=http://127.0.0.1:10808 HTTP_PROXY=http://127.0.0.1:10808 HF_HUB_DISABLE_XET=1
while true; do
 count=$(find cache/adv_images/DHRA -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)
 if [ "$count" -ge 200 ]; then break; fi
 if ! screen -ls | grep -q hattack200; then echo "200 run stopped at $count" >> /root/autodl-tmp/hattack1000.log; exit 1; fi
 sleep 60
done
python -u prepare_1000.py > /root/autodl-tmp/hattack1000.log 2>&1 || exit 1
sed -i 's/num_samples = 200/num_samples = 1000/' config.toml
for attempt in $(seq 1 5); do
 echo "ATTEMPT $attempt $(date -Is)" >> /root/autodl-tmp/hattack1000.log
 python -u gen_adv.py >> /root/autodl-tmp/hattack1000.log 2>&1
 code=$?
 count=$(find cache/adv_images/DHRA -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)
 echo "EXIT $code OUTPUT_COUNT $count $(date -Is)" >> /root/autodl-tmp/hattack1000.log
 if [ "$code" -eq 0 ] && [ "$count" -ge 1000 ]; then exit 0; fi
 sleep 10
done
exit 1
