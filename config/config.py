import toml
from pathlib import Path
from pydantic import BaseModel

class Data(BaseModel):
    num_samples: int = 1000
    src_data_path: str = "res/source_1000"
    tgt_data_path: str = "res/target_1000"

class Optim(BaseModel):
    alpha: float = 1.0
    epsilon: int = 16
    steps: int = 300

class Model(BaseModel):
    resize: int = 224
    use_source_crop: bool = True
    use_target_crop: bool = True
    crop_scale: tuple = (0.5, 0.9)
    backbones: list = ("B16", "B32", "Laion")

class Blackbox(BaseModel):
    method: str = "FOA"
    model_name: list | str = "gpt4o"

class Config(BaseModel):
    method: str
    blackbox: list | str
    device: str = "cuda:0"
    attack: str = "fgsm"
    data: Data
    optim: Optim
    model: Model

class Apikey(BaseModel):
    normal: str
    claude: str
    gemini: str
    gpt: str

def TomlLoader(path: Path, cls):
    with open(path, "r", encoding="utf-8") as f:
        return cls(**toml.load(f))