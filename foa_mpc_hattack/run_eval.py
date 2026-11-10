"""Evaluate generated attacks with Hattack's fixed local caption/judge pipeline."""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

MODELS = ("qwen3b", "qwen7b", "llava15", "llava16")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", required=True, choices=("E0", "E1", "E1LDI", "E2", "E3", "E4", "E4NoMI", "E4NoMIFOA", "E4LDI", "E4LDINoMIFOA", "E3LDI_C01", "E3LDI_C001", "E3LDI_C1", "E3LDI_C0001"))
    p.add_argument("--num-samples", type=int, required=True)
    p.add_argument("--seed", type=int, default=2023)
    p.add_argument("--out-dir", type=Path, default=Path("cache/foa_mpc_hattack"))
    p.add_argument("--target-dir", type=Path, default=Path("datas/coco_val2017_1000"))
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    args = p.parse_args()
    source = (args.out_dir / f"{args.variant}_mpc_official" / "img" / "images"
              if args.variant in ("E4", "E4NoMI", "E4NoMIFOA", "E4LDI", "E4LDINoMIFOA") else args.out_dir / "img" / args.variant)
    method = f"FOA_MPC_{args.variant}_N{args.num_samples}"
    destination = Path("cache/adv_images") / method
    destination.mkdir(parents=True, exist_ok=True)
    for i in range(args.num_samples):
        name = f"{i:04d}.png"
        image = source / name
        if not image.is_file() or not (args.target_dir / name).is_file():
            raise FileNotFoundError(f"Missing generated/target pair {i}: {image}")
        link = destination / name
        if not link.exists():
            link.symlink_to(image.resolve())
    for model in args.models:
        for script in ("caption_open_models.py", "score_open_captions.py"):
            command = [sys.executable, script, "--model", model, "--method", method,
                       "--limit", str(args.num_samples)]
            if script == "caption_open_models.py":
                command += ["--target-dir", str(args.target_dir)]
            print("RUN", " ".join(command), flush=True)
            subprocess.run(command, check=True)
        records_path = Path("cache/evaluation") / method / model / "judge_local_qwen7b" / "records.json"
        records = json.loads(records_path.read_text(encoding="utf-8"))
        output = args.out_dir / f"{args.variant}_{model}_paired_scores.csv"
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("source_id", "target_id", "victim_id", "score", "success", "seed"))
            for i in range(args.num_samples):
                key = f"{i:04d}"
                score = float(records[key]["similarity"])
                target_id = Path(records[key]["target"]).stem
                seed = args.seed if args.variant in ("E4", "E4NoMI", "E4NoMIFOA", "E4LDI", "E4LDINoMIFOA") else args.seed + i
                writer.writerow((key, target_id, model, score, int(score >= 0.5), seed))
        print("SCORES", output, flush=True)


if __name__ == "__main__":
    main()
