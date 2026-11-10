"""Download the exact public checkpoints used by the FOA/MPC repositories.

Run on AutoDL with HF_HOME on the data disk. Completed snapshots are skipped.
"""
import os
import time
from huggingface_hub import snapshot_download

MODELS = (
    "openai/clip-vit-base-patch16",
    "openai/clip-vit-base-patch32",
    "laion/CLIP-ViT-G-14-laion2B-s12B-b42K",
    "facebook/dinov2-base",
    "OpenGVLab/InternVL3-1B-hf",
)

os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")
for model_id in MODELS:
    for attempt in range(1, 21):
        try:
            print("DOWNLOAD", model_id, "ATTEMPT", attempt, flush=True)
            path = snapshot_download(
                model_id, max_workers=2,
                allow_patterns=["model*.safetensors", "*.json", "*.txt", "*.model",
                                "vocab.*", "merges.txt", "tokenizer.*", "*.py", "*.jinja"],
                ignore_patterns=["*.bin", "*.h5", "*.msgpack", "*.onnx"])
            print("COMPLETE", model_id, path, flush=True)
            break
        except Exception as exc:
            print("RETRY", model_id, repr(exc), flush=True)
            time.sleep(10)
    else:
        raise RuntimeError(f"Could not download {model_id} after 20 attempts")
