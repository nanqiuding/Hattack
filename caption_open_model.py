"""Generate paired captions with a locally hosted open-source VLM.

Writes records compatible with evaluate_attack.py, which can later add GPTScore.
"""
import argparse
import json
import os
from pathlib import Path

PROMPT = "Describe this image in one concise sentence, no longer than 20 words."


def save(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--judge", default="gpt-4o")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    adv_dir = Path("cache/adv_images") / args.method
    adv_paths = sorted(adv_dir.glob("*.png"))
    if args.limit:
        adv_paths = adv_paths[:args.limit]
    if not adv_paths:
        parser.error(f"No adversarial images in {adv_dir}")
    pairs = [(adv, args.target_dir / adv.name) for adv in adv_paths]
    missing = [str(tgt) for _, tgt in pairs if not tgt.is_file()]
    if missing:
        parser.error(f"Missing {len(missing)} targets, first: {missing[0]}")
    model_slug = args.model.split("/")[-1]
    state = Path("cache/evaluation") / args.method / model_slug / f"judge_{args.judge}" / "records.json"
    records = json.loads(state.read_text(encoding="utf-8")) if state.exists() else {}
    print(f"Validated {len(pairs)} pairs; output={state}", flush=True)
    if args.dry_run:
        return

    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="cuda:0"
    ).eval()

    def caption(path):
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": PROMPT}
        ]}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            inputs = processor(text=[text], images=[image], padding=True,
                               return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        generated = output[:, inputs["input_ids"].shape[1]:]
        result = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        if not result:
            raise RuntimeError(f"Empty caption for {path}")
        return result

    for index, (adv, target) in enumerate(pairs, 1):
        item = records.setdefault(adv.stem, {
            "adversarial": str(adv), "target": str(target)
        })
        if item["adversarial"] != str(adv) or item["target"] != str(target):
            raise RuntimeError(f"Pair changed for {adv.stem}")
        for field, path in (("target_description", target),
                            ("adversarial_description", adv)):
            if not item.get(field):
                item[field] = caption(path)
                save(state, records)
        print(f"{index}/{len(pairs)} {adv.stem}: {item['adversarial_description']}", flush=True)


if __name__ == "__main__":
    main()
