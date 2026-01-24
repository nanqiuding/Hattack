import random
from tqdm import tqdm
from pathlib import Path

from utils import ImageDescriptionGeneratorClose, write_json, read_json

# model_name = ["claude-sonnet-4-5", "gemini-2.0-flash", "gpt-4o", "gpt-4.1"]
# method_name = ["AdvDiffVLM", "SSA-CWA", "AnyAttack", "M-Attack"]
model_name = ["o3", "gemini-2.5-pro", "claude-sonnet-4-5"]
method_name = ["AdvDiffVLM", "SSA-CWA"]

gener = [ImageDescriptionGeneratorClose(name, thinking=True) for name in model_name]

def get_image_paths(root_path: Path):
    return sorted(root_path.iterdir(), key=lambda p: int(p.stem))

random.seed(2026)
# img_ids = random.sample(range(1000), 100)
img_ids = list(range(0, 1000, 10))
# img_ids = list(range(100))

for method in method_name:
    adv_path = Path("cache/adv_images") / method
    desc_path = Path("cache/adv_description") / method / "thinking"
    desc_path.mkdir(parents=True, exist_ok=True)

    adv_paths = get_image_paths(adv_path)

    for name, gen in zip(model_name, gener):
        descriptions = []
        if (desc_path / f"{name}.json").exists():
            descriptions = read_json(desc_path / f"{name}.json")

        for i, id in enumerate(tqdm(img_ids, desc=f"{method}, {name}")):
            if i < len(descriptions):
                continue
            path = adv_paths[id]
        # for path in tqdm(adv_paths, desc=f"{method}, {name}"):
            desc = {}
            desc["path"] = str(path)
            desc["description"] = gen.generate_description(path)
            descriptions.append(desc)

            write_json(descriptions, desc_path / f"{name}.json")