from tqdm import tqdm
from pathlib import Path

from utils import ImageDescriptionGeneratorClose, write_json, read_json

# model_name = ["claude-sonnet-4-5", "gemini-2.0-flash", "gpt-4o", "gpt-4.1"]
model_name = ["gemini-2.0-flash"]
method_name = [
    "FOA_inter-30-intra_2", "FOA_inter-30-intra_4", "FOA_inter-30-intra_8"
]

gener = [ImageDescriptionGeneratorClose(name) for name in model_name]

def get_image_paths(root_path: Path):
    return (
        sorted((root_path / "cluster_3").iterdir(), key=lambda p: int(p.stem)),
        sorted((root_path / "cluster_5").iterdir(), key=lambda p: int(p.stem))
    )

img_ids = list(range(0, 1000, 10))

for method in method_name:
    adv_path = Path("cache/adv_images") / method
    desc_path = Path("cache/adv_description") / method
    desc_path.mkdir(parents=True, exist_ok=True)
    
    img_path_3, img_path_5 = get_image_paths(adv_path)
    
    for name, gen in zip(model_name, gener):
        print(method, name)

        descriptions = []
        save_path = desc_path / f"{name}.json"
        if save_path.exists():
            descriptions = read_json(save_path)

        # for id in tqdm(img_ids):
        #     path_3, path_5 = img_path_3[id], img_path_5[id]
        for i, (path_3, path_5) in enumerate(zip(tqdm(img_path_3), img_path_5)):
            if i < len(descriptions):
                continue

            desc = {}
            desc["path_cluster_3"] = str(path_3)
            desc["description_cluster_3"] = gen.generate_description(path_3)
            desc["path_cluster_5"] = str(path_5)
            desc["description_cluster_5"] = gen.generate_description(path_5)
            descriptions.append(desc)

            write_json(descriptions, save_path)