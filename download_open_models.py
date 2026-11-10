"""Resume downloads of the four ungated open-model checkpoints."""
import os
import shutil
import time
from pathlib import Path
from huggingface_hub import snapshot_download

os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:10808'
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:10808'
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ.pop('HF_HUB_OFFLINE', None)
os.environ.pop('TRANSFORMERS_OFFLINE', None)
models = [
    ('Qwen/Qwen2.5-VL-3B-Instruct', None, 8),
    ('Qwen/Qwen2.5-VL-7B-Instruct', '/root/autodl-tmp/vlm_models/Qwen2.5-VL-7B-Instruct', 17),
    ('llava-hf/llava-1.5-7b-hf', '/root/autodl-tmp/vlm_models/llava-1.5-7b-hf', 15),
    ('llava-hf/llava-v1.6-vicuna-7b-hf', '/root/autodl-tmp/vlm_models/llava-v1.6-vicuna-7b-hf', 15),
]
for repo, location, size_gb in models:
    target = Path(location or '/root/.cache/huggingface')
    target.mkdir(parents=True, exist_ok=True)
    available = shutil.disk_usage(target).free / 1e9
    print('START', repo, 'free_GB', round(available, 2), flush=True)
    if available < size_gb + 2 and not (location is None and available >= 5):
        print('INSUFFICIENT_SPACE', repo, flush=True)
        break
    for attempt in range(1, 21):
        try:
            path = snapshot_download(repo_id=repo, local_dir=location,
                max_workers=2, ignore_patterns=['*.bin','*.h5','*.msgpack','*.gguf'])
            print('COMPLETE', repo, path, flush=True)
            break
        except Exception as exc:
            print('ERROR', repo, attempt, repr(exc), flush=True)
            if attempt == 20: raise
            time.sleep(20)
print('DOWNLOAD_LOOP_DONE', flush=True)
