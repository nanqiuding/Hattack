import json
import random
from pathlib import Path

def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# model_name = ["claude-sonnet-4-5", "gemini-2.0-flash", "gpt-4o", "gpt-4.1"]
model_name = ["gemini-2.0-flash"]
method_name = [
    "FOA_inter-30-intra_2", "FOA_inter-30-intra_4", "FOA_inter-30-intra_8"
]

res_path = Path("cache/results")

random.seed(2026)
# img_ids = random.sample(range(1000), 100)
img_ids = list(range(0, 1000, 10))

for method in method_name:
    for model in model_name:
        scores = read_json(res_path / method / f"{model}.json")
        success = 0
        score = 0.0
        # for id in img_ids:
        #     s = scores[id]
        for s in scores:
            s_3 = s.get("similary_3", 0.0)
            s_5 = s.get("similary_5", 0.0)
            s_ = max(s_3, s_5)
            success += s_ >= 0.5
            score += s_
        print(f"{method}, {model}:")
        print(success, f"{score / len(img_ids):.3f}")