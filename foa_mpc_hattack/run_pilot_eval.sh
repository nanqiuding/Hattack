#!/usr/bin/env bash
set -u
cd /root/Hattack
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
while screen -ls | grep -Eq 'foa_mpc_pilot|foa_mpc_e4'; do sleep 30; done
out=/root/Hattack/cache/foa_mpc_hattack/pilot5
for variant in E0 E1 E2 E4; do
  if [ "$variant" = E4 ]; then image_dir="$out/E4_mpc_official/img/images";
  else image_dir="$out/img/$variant"; fi
  count=$(find "$image_dir" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)
  if [ "$count" -ne 5 ]; then
    echo "EVAL_SKIPPED $variant images=$count"
    continue
  fi
  echo "EVAL_START $variant $(date -Is)"
  /root/miniconda3/bin/python -u -m foa_mpc_hattack.run_eval \
    --variant "$variant" --num-samples 5 --out-dir "$out"
  echo "EVAL_EXIT $variant $? $(date -Is)"
done
/root/miniconda3/bin/python -m foa_mpc_hattack.summarize --out-dir "$out"
