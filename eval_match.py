from pathlib import Path

from utils import read_json

model_name = ["claude-sonnet-4-5", "gemini-2.0-flash", "gpt-4o", "gpt-4.1"]
method_name = ["AdvDiffVLM", "SSA-CWA", "AnyAttack", "M-Attack"]

match_path = Path("cache/matchs")

for method in method_name:
    for model in model_name:
        matchs = read_json(match_path / method / f"{model}.json")

        kmr = [0, 0, 0]

        for m in matchs:
            match_num = len(m["match"])
            if match_num > 2:
                kmr[2] += 1
            elif match_num > 1:
                kmr[1] += 1
            elif match_num > 0:
                kmr[0] += 1
            
        print(f"{method}, {model}: KMR_a {kmr[0]}, KMR_b {kmr[1]}, KMR_c {kmr[2]}")