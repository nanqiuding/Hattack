"""Score paired local-VLM captions with one fixed local Qwen2.5-VL-7B judge."""
import argparse
import json
import os
import re
from pathlib import Path

from utils.scorer import scorer_prompt

JUDGE_PATH = "/root/autodl-tmp/vlm_models/Qwen2.5-VL-7B-Instruct"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=("qwen3b", "qwen7b", "llava15", "llava16"))
    parser.add_argument("--method", default="DHRA_NIPS1000")
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()
    output = Path("cache/evaluation") / args.method / args.model / "judge_local_qwen7b"
    records_path = output / "records.json"
    records = json.loads(records_path.read_text(encoding="utf-8"))
    image_ids = [f"{i:04d}" for i in range(args.limit)]
    if any(i not in records or not records[i].get("target_description") or
           not records[i].get("adversarial_description") for i in image_ids):
        parser.error("All paired captions must exist before scoring")

    import torch
    from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration
    tokenizer = AutoTokenizer.from_pretrained(JUDGE_PATH, local_files_only=True)
    judge = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        JUDGE_PATH, torch_dtype=torch.float16, device_map="cuda:0", local_files_only=True
    ).eval()
    print("JUDGE_READY", args.model, args.limit, flush=True)

    def score(first, second):
        prompt = scorer_prompt.format(first, second)
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokens = tokenizer(text, return_tensors="pt").to("cuda:0")
        with torch.inference_mode():
            generated = judge.generate(**tokens, max_new_tokens=16, do_sample=False)
        raw = tokenizer.decode(generated[0, tokens["input_ids"].shape[1]:],
                               skip_special_tokens=True).strip()
        match = re.search(r"(?<!\d)(?:0(?:\.\d+)?|1(?:\.0+)?)(?!\d)", raw)
        if match is None:
            raise ValueError(f"Judge returned no valid 0–1 score: {raw!r}")
        return float(match.group()), raw

    for index, image_id in enumerate(image_ids, 1):
        item = records[image_id]
        if "similarity" not in item:
            item["similarity"], item["judge_raw"] = score(
                item["target_description"], item["adversarial_description"])
            save(records_path, records)
        if index % 20 == 0 or index == args.limit:
            print("SCORE_PROGRESS", args.model, index, args.limit, flush=True)
    values = [records[i]["similarity"] for i in image_ids]
    summary = {
        "method": args.method, "caption_model": args.model,
        "judge": "local Qwen2.5-VL-7B-Instruct (text-only)",
        "not_comparable_to_paper_gpt4o_judge": True,
        "samples": len(values), "threshold": 0.5,
        "successes": sum(x >= 0.5 for x in values),
        "ASR": sum(x >= 0.5 for x in values) / len(values),
        "AvgSim": sum(values) / len(values),
    }
    save(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
