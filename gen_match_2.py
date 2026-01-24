from tqdm import tqdm
from pathlib import Path

from utils import KeywordMatcher, read_json, write_json

matcher = KeywordMatcher(model_name="gpt-4o")

model_name = ["claude-sonnet-4-5", "gemini-2.0-flash", "gpt-4o", "gpt-4.1"]
method_name = ["FOA", "FOA_inter-30-intra"]

score_path = Path("cache/results")
match_path = Path("cache/matchs")

keywords = read_json("datas/keywords.json")

for method in method_name:
    res_match_path = match_path / method
    res_match_path.mkdir(parents=True, exist_ok=True)

    for model in model_name:
        scores = read_json(score_path / method / f"{model}.json")
        matchs = []
        for id in tqdm(range(100), desc=f"{method}, {model}"):
            res = {}
            res["path"] = scores[id]["path_adv_3"]
            res["description_target"] = scores[id]["description_target"]
            res["description_adv"] = scores[id]["description_adv_3"]
            res["keywords"] = keywords[id]["keywords"]
            
            if "similary_5" in scores[id] and scores[id]["similary_5"] >= scores[id]["similary_3"]:
                res["path"] = scores[id]["path_adv_5"]
                res["description_adv"] = scores[id]["description_adv_5"]
            
            res["match"] = matcher.match(
                desc=res["description_adv"],
                keywords=res["keywords"]
            )

            matchs.append(res)
            write_json(matchs, res_match_path / f"{model}.json")