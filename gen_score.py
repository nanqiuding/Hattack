from tqdm import tqdm
from pathlib import Path

from utils import GPTScorer, read_json, write_json

scorer = GPTScorer(model_name="gpt-4o")
success_threshold = 0.5

# model_name = ["claude-sonnet-4-5", "gemini-2.0-flash", "gpt-4o", "gpt-4.1"]
model_name = ["o3", "gemini-2.5-pro", "claude-sonnet-4-5"]
method_name = ["AdvDiffVLM", "SSA-CWA"]

adv_desc_path = Path("cache/adv_description")
target_desc_path = Path("cache/target_description")
result_path = Path("cache/results")

for method in method_name:
    res_method_path = result_path / method / "thinking"
    res_method_path.mkdir(parents=True, exist_ok=True)
    for model in model_name:
        target = read_json(target_desc_path / "thinking" / f"{model}.json")
        adv = read_json(adv_desc_path / method / "thinking" / f"{model}.json")

        result = []
        # for a in tqdm(adv, desc=f"{method}, {model}"):
        #     id = int(Path(a["path"]).stem)
        #     t = target[id]
        for a, t in zip(tqdm(adv, desc=f"{method}, {model}"), target):
            
            res = {}
            res["path_target"] = t["path"]
            res["description_target"] = t["description"]
            res["path_adv"] = a["path"]
            res["description_adv"] = a["description"]
            
            sim = scorer.compute_similarity(
                t["description"],
                a["description"]
            )
            res["similary"] = sim

            result.append(res)
            write_json(result, res_method_path / f"{model}.json")