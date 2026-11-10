"""Summarize identical sample IDs across E0/E1/E2/E4 local evaluations."""
import argparse
import csv
import json
from pathlib import Path

VARIANTS = ("E0", "E1", "E1LDI", "E2", "E3", "E4", "E4NoMI", "E4NoMIFOA", "E4LDI", "E3LDI_C01", "E3LDI_C001", "E3LDI_C1", "E3LDI_C0001")


def read_scores(path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    result = {(row["source_id"], row["target_id"]): float(row["score"]) for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate source/target pair in {path}")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=Path("cache/foa_mpc_hattack"))
    p.add_argument("--models", nargs="+", default=["qwen3b", "qwen7b", "llava15", "llava16"])
    args = p.parse_args()
    result = {}
    for model in args.models:
        available = {}
        for variant in VARIANTS:
            path = args.out_dir / f"{variant}_{model}_paired_scores.csv"
            if path.is_file():
                available[variant] = read_scores(path)
        if not available:
            continue
        common = set.intersection(*(set(scores) for scores in available.values()))
        if not common:
            raise ValueError(f"No common image pairs for {model}")
        model_result = {"samples": len(common), "variants": {}, "E2_vs_E1": None}
        for variant, scores in available.items():
            values = [scores[pair] for pair in sorted(common)]
            model_result["variants"][variant] = {
                "ASR": sum(x >= 0.5 for x in values) / len(values),
                "AvgSim": sum(values) / len(values)}
        if "E1" in available and "E2" in available:
            e1, e2 = available["E1"], available["E2"]
            model_result["E2_vs_E1"] = {
                "win": sum(e2[p] >= 0.5 > e1[p] for p in common),
                "loss": sum(e1[p] >= 0.5 > e2[p] for p in common),
                "tie": sum((e1[p] >= 0.5) == (e2[p] >= 0.5) for p in common)}
        result[model] = model_result
    output = args.out_dir / "paired_summary.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
