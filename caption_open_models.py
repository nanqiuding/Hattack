"""Caption target/adversarial pairs with one local VLM; resume per caption."""
import argparse
import json
import os
from pathlib import Path

PROMPT = "Describe this image in one concise sentence, no longer than 20 words."
MODELS = {
    "qwen3b": ("Qwen/Qwen2.5-VL-3B-Instruct", "qwen"),
    "qwen7b": ("/root/autodl-tmp/vlm_models/Qwen2.5-VL-7B-Instruct", "qwen"),
    "llava15": ("/root/autodl-tmp/vlm_models/llava-1.5-7b-hf", "llava"),
    "llava16": ("/root/autodl-tmp/vlm_models/llava-v1.6-vicuna-7b-hf", "llava_next"),
}


def save(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--method", default="DHRA_NIPS1000")
    parser.add_argument("--target-dir", type=Path, default=Path("datas/coco_val2017_1000"))
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()
    paths = sorted((Path("cache/adv_images") / args.method).glob("*.png"))[:args.limit]
    if len(paths) != args.limit:
        parser.error(f"Expected {args.limit} adversarial images; found {len(paths)}")
    if any(not (args.target_dir / p.name).is_file() for p in paths):
        parser.error("Missing paired target image")
    state = Path("cache/evaluation") / args.method / args.model / "judge_local_qwen7b" / "records.json"
    records = json.loads(state.read_text(encoding="utf-8")) if state.exists() else {}

    import torch
    from PIL import Image
    from huggingface_hub import snapshot_download
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, LlavaForConditionalGeneration, LlavaNextForConditionalGeneration

    source, kind = MODELS[args.model]
    if args.model == "qwen3b":
        source = snapshot_download(source, local_files_only=True)
    processor = AutoProcessor.from_pretrained(source, local_files_only=True)
    model_class = {"qwen": Qwen2_5_VLForConditionalGeneration,
                   "llava": LlavaForConditionalGeneration,
                   "llava_next": LlavaNextForConditionalGeneration}[kind]
    model = model_class.from_pretrained(source, torch_dtype=torch.float16,
                                        device_map="cuda:0", local_files_only=True).eval()
    conversation = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": PROMPT}
    ]}]
    prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    print("MODEL_READY", args.model, "PAIRS", len(paths), flush=True)

    def caption(path):
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            inputs = processor(text=[prompt], images=[image], padding=True, return_tensors="pt")
        inputs = inputs.to("cuda:0", dtype=torch.float16)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        result = processor.batch_decode(generated[:, inputs["input_ids"].shape[1]:],
                                        skip_special_tokens=True)[0].strip()
        if not result:
            raise RuntimeError(f"Empty caption for {path}")
        return result

    for index, adv in enumerate(paths, 1):
        target = args.target_dir / adv.name
        item = records.setdefault(adv.stem, {"adversarial": str(adv), "target": str(target)})
        if item["adversarial"] != str(adv) or item["target"] != str(target):
            raise RuntimeError(f"Image pair changed: {adv.stem}")
        for field, path in (("target_description", target), ("adversarial_description", adv)):
            if not item.get(field):
                item[field] = caption(path)
                save(state, records)
        if index % 10 == 0 or index == len(paths):
            print("PROGRESS", args.model, index, len(paths), flush=True)
    print("CAPTIONS_COMPLETE", args.model, state, flush=True)


if __name__ == "__main__":
    main()
