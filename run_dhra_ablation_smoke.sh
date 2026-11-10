#!/usr/bin/env bash
set -e
cd /root/Hattack
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
for v in A0 A1 A2 A3 B1 B2 B3 C1 C2; do
 echo SMOKE_START "$v" $(date -Is)
 /root/miniconda3/bin/python -u dhra_ablation.py --variant "$v" --limit 1 --steps 2
 echo SMOKE_DONE "$v" $(date -Is)
done
