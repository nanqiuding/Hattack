"""Print completed three-model score summaries without changing experiment files."""
import csv
from pathlib import Path

base = Path("/root/Hattack/cache/foa_mpc_hattack/stage100")
variants = ("E1LDI", "E4LDI", "E3LDI_C01", "E3LDI_C001", "E3LDI_C1", "E3LDI_C0001")
models = ("qwen3b", "qwen7b", "llava15", "llava16")
for variant in variants:
    ledger = base / f"{variant}_samples.jsonl"
    count = sum(1 for _ in ledger.open(encoding="utf-8")) if ledger.exists() else None
    if count is None:
        count = len(list((base / f"{variant}_mpc_official/img/images").glob("*.png")))
    print(variant, "generated", count, flush=True)
    for model in models:
        path = base / f"{variant}_{model}_paired_scores.csv"
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as stream:
            values = [float(row["score"]) for row in csv.DictReader(stream)]
        print(model, len(values), f"ASR={sum(x >= 0.5 for x in values) / len(values):.3f}",
              f"AvgSim={sum(values) / len(values):.3f}", flush=True)
