#!/usr/bin/env bash
export HTTPS_PROXY=http://127.0.0.1:10808 HTTP_PROXY=http://127.0.0.1:10808 HF_HUB_DISABLE_XET=1
for attempt in $(seq 1 20); do
 echo "ATTEMPT $attempt $(date -Is)" >> /root/autodl-tmp/qwen3b_download.log
 python - <<'PY' >> /root/autodl-tmp/qwen3b_download.log 2>&1
from huggingface_hub import snapshot_download
p=snapshot_download('Qwen/Qwen2.5-VL-3B-Instruct',max_workers=1)
print('COMPLETE',p,flush=True)
PY
 if [ "$?" -eq 0 ]; then exit 0; fi
 sleep 10
done
exit 1
