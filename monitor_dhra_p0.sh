#!/usr/bin/env bash
cd /root/Hattack
while ! grep -q '^STAGE20_COMPLETE' /root/autodl-tmp/dhra_p0.log; do
 if ! screen -ls | grep -q dhra_p0; then echo 'P0 runner exited before stage20 complete'; exit 1; fi
 sleep 60
done
/root/miniconda3/bin/python build_dhra_ablation_report.py --stage stage20
while ! grep -q '^P0_COMPLETE' /root/autodl-tmp/dhra_p0.log; do
 if ! screen -ls | grep -q dhra_p0; then echo 'P0 runner exited before full complete'; exit 1; fi
 sleep 300
done
/root/miniconda3/bin/python build_dhra_ablation_report.py --stage full
