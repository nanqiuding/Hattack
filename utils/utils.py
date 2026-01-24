import os
import json
import torch
import base64
import random
import numpy as np
from PIL import Image
from pathlib import Path

from config import TomlLoader, Apikey

def set_seed(seed=2023):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_api_keys() -> Apikey:
    return TomlLoader("api_keys.toml", Apikey)

def get_media_type(image_path: Path) -> str:
    ext = image_path.suffix
    if ext in [".jpg", ".jpeg", ".JPEG"]:
        return "image/jpeg"
    elif ext == ".png":
        return "image/png"
    else:
        raise ValueError(f"Unsupported image extension: {ext}")

def encode_image(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def to_tensor(pic: Image):
    mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
    img = torch.from_numpy(
        np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True)
    )
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    img = img.permute((2, 0, 1)).contiguous()
    return img.to(dtype=torch.get_default_dtype())

def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(data, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)