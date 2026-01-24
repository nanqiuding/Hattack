import random
from tqdm import tqdm
from pathlib import Path

from utils import ImageDescriptionGeneratorClose, write_json

# model_name = ["claude-sonnet-4-5"]
model_name = ["claude-sonnet-4-5"]

gener = [ImageDescriptionGeneratorClose(name, thinking=True) for name in model_name]

def get_image_paths(root_path: Path):
    return sorted(root_path.iterdir(), key=lambda p: int(p.stem))

random.seed(2026)
# img_ids = random.sample(range(1000), 100)
img_ids = list(range(0, 1000, 10))

tgt_path = Path("datas/target_1000")
desc_path = Path("cache/target_description") / "thinking"
desc_path.mkdir(parents=True, exist_ok=True)

adv_paths = get_image_paths(tgt_path)

for name, gen in zip(model_name, gener):
    descriptions = []
    for id in tqdm(img_ids, desc=name):
        desc = {}
        path = adv_paths[id]
        desc["path"] = str(path)
        desc["description"] = gen.generate_description(path)
        descriptions.append(desc)

        write_json(descriptions, desc_path / f"{name}.json")