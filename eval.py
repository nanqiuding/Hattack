import json
import random
from pathlib import Path

def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

model_name = ["o3", "gemini-2.5-pro", "claude-sonnet-4-5"]
method_name = ["AdvDiffVLM", "SSA-CWA"]

res_path = Path("cache/results")

random.seed(2026)
# img_ids = random.sample(range(1000), 100)
img_ids = list(range(0, 1000, 10))

for method in method_name:
    for model in model_name:
        scores = read_json(res_path / method / "thinking" / f"{model}.json")
        success = 0
        score = 0.0
        # for id in img_ids:
        #     s = scores[id]
        for s in scores:
            s_ = s.get("similary", 0.0)
            success += s_ >= 0.5
            score += s_
        print(f"{method}, {model}:")
        print(success, f"{score / len(img_ids):.3f}")