from tqdm import tqdm
from pathlib import Path

from utils import GPTScorer, read_json, write_json

scorer = GPTScorer(model_name="gpt-4o")
success_threshold = 0.5

# model_name = ["claude-sonnet-4-5", "gemini-2.0-flash", "gpt-4o", "gpt-4.1"]
model_name = ["gemini-2.0-flash"]
method_name = [
    "FOA_inter-30-intra_2", "FOA_inter-30-intra_4", "FOA_inter-30-intra_8"
]

adv_desc_path = Path("cache/adv_description")
target_desc_path = Path("cache/target_description")
result_path = Path("cache/results")

for method in method_name:
    res_method_path = result_path / method
    res_method_path.mkdir(parents=True, exist_ok=True)
    for model in model_name:
        target = read_json(target_desc_path / f"{model}.json")
        adv = read_json(adv_desc_path / method / f"{model}.json")

        result = []
        for a in tqdm(adv, desc=f"{method}, {model}"):
            id = int(Path(a["path_cluster_3"]).stem)
            t = target[id]

            res = {}
            res["path_target"] = t["path"]
            res["description_target"] = t["description"]

            sim_3 = scorer.compute_similarity(
                t["description"],
                a["description_cluster_3"]
            )
            res["path_adv_3"] = a["path_cluster_3"]
            res["description_adv_3"] = a["description_cluster_3"]
            res["similary_3"] = sim_3

            if sim_3 < success_threshold:
                sim_5 = scorer.compute_similarity(
                    t["description"],
                    a["description_cluster_5"]
                )            
                res["path_adv_5"] = a["path_cluster_5"]
                res["description_adv_5"] = a["description_cluster_5"]
                res["similary_5"] = sim_5

            result.append(res)
            write_json(result, res_method_path / f"{model}.json")
