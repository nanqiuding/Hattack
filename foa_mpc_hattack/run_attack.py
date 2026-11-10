"""Run matched FOA / FOA+MPC / FOA+MPC+Fidelity attacks.

E0, E1 and E2 call FOA's original clustered-OT and dynamic-weight classes.
E4 is dispatched to the vendored MPC source by run_mpc_original.py.
"""
import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from .fidelity import fidelity_loss
CLIP_NAMES = ("B16", "B32", "Laion")
COH_LDI = {"E3LDI_C01": 0.1, "E3LDI_C001": 0.01,
           "E3LDI_C1": 1.0, "E3LDI_C0001": 0.001}
VARIANTS = ("E0", "E1", "E1LDI", "E2", "E3", "E4", "E4NoMI", "E4NoMIFOA", "E4LDI", "E4LDINoMIFOA", *COH_LDI)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_pairs(data_root, n):
    data_root = Path(data_root)
    src = sorted((data_root / "nips2017_1000").rglob("*.png"))
    tgt = sorted((data_root / "coco_val2017_1000").rglob("*.png"))
    if len(src) < n or len(tgt) < n:
        raise ValueError(f"Need {n} paired files, found {len(src)} source and {len(tgt)} target")
    return list(zip(src[:n], tgt[:n]))


def load_image(path):
    with Image.open(path) as im:
        im = im.convert("RGB")
        im = transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC)(im)
        im = transforms.CenterCrop(224)(im)
        return transforms.ToTensor()(im).unsqueeze(0) * 255.0


def cls_layers(model, image_255):
    pixels = model.normalizer(image_255)
    states = model.model.vision_model(pixel_values=pixels, output_hidden_states=True).hidden_states
    return torch.stack([state[:, 0, :] for state in states[1:]], 1)


def run_foa_family(args, pairs, out_dir):
    from .adapters import DINOv2Adapter, InternVL3Adapter
    from .vendor.foa_official.surrogates.FeatureExtractors.Base import (
        EnsembleFeatureExtractor_ot, EnsembleFeatureLoss_OT_foa_attack)
    from .vendor.foa_official.surrogates.FeatureExtractors.ClipB16 import ClipB16FeatureExtractor
    from .vendor.foa_official.surrogates.FeatureExtractors.ClipB32 import ClipB32FeatureExtractor
    from .vendor.foa_official.surrogates.FeatureExtractors.ClipLaion import ClipLaionFeatureExtractor
    clip_types = {"B16": ClipB16FeatureExtractor, "B32": ClipB32FeatureExtractor,
                  "Laion": ClipLaionFeatureExtractor}
    device = torch.device(args.device)
    set_seed(args.seed)
    models = [clip_types[name]().eval().to(device).requires_grad_(False)
              for name in args.clip_backbones]
    extractor = EnsembleFeatureExtractor_ot(models, cluster_number=10)
    foa_loss = EnsembleFeatureLoss_OT_foa_attack(models, cluster_number=10)
    extras = []
    if args.variant in ("E1", "E1LDI", "E2", "E3", *COH_LDI):
        extras = [("DINOv2", DINOv2Adapter(device).eval()),
                  ("InternVL3_1B", InternVL3Adapter(device).eval())]
    for _, model in extras:
        model.requires_grad_(False)
    source_crop = transforms.RandomResizedCrop(224, scale=(0.5, 0.9))
    target_crop = transforms.RandomResizedCrop(224, scale=(0.5, 0.9))
    image_dir = out_dir / "img" / args.variant
    image_dir.mkdir(parents=True, exist_ok=True)
    ledger = out_dir / f"{args.variant}_samples.jsonl"
    completed = {}
    if ledger.exists():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            completed[item["source_id"]] = item
    for index, (src_path, tgt_path) in enumerate(pairs):
        output_path = image_dir / src_path.name
        if output_path.exists() and not args.overwrite:
            record = completed.get(src_path.stem)
            if record is None or record.get("target_id") != tgt_path.stem:
                raise RuntimeError(f"Cannot resume {output_path}: missing/mismatched ledger record")
            foa_loss.previous_loss_list = [torch.tensor(x, device=device)
                                           for x in record["foa_previous_losses"]]
            print("SKIP", args.variant, index, output_path, flush=True)
            continue
        sample_seed = args.seed + index
        set_seed(sample_seed)
        src, tgt = load_image(src_path).to(device), load_image(tgt_path).to(device)
        delta = torch.zeros_like(src)
        start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        last_metrics = {}
        for step in range(args.steps):
            delta = delta.detach().requires_grad_(True)
            target_view = target_crop(tgt)
            with torch.no_grad():
                foa_loss.set_ground_truth(target_view)
            adv_view = source_crop(src + delta)
            global_feats, local_feats = extractor(adv_view)
            foa_score = foa_loss(global_feats, local_feats)
            score = foa_score
            extra_scores = {}
            for name, adapter in extras:
                with torch.no_grad():
                    target_feat = adapter.forward_features(target_view / 255).global_feat
                adv_feat = adapter.forward_features(adv_view / 255).global_feat
                extra_scores[name] = F.cosine_similarity(adv_feat, target_feat, dim=-1).mean()
            if extra_scores:
                score = score + args.extra_global_weight * torch.stack(list(extra_scores.values())).mean()
            coh = None
            if args.variant in ("E2", "E3", *COH_LDI) and args.lambda_coh > 0:
                with torch.no_grad():
                    target_layers = [cls_layers(model, target_view) for model in models]
                adv_layers = [cls_layers(model, adv_view) for model in models]
                coh_terms = [fidelity_loss(a, t) for a, t in zip(adv_layers, target_layers)]
                if args.variant == "E3" or args.variant in COH_LDI:
                    for _, adapter in extras:
                        with torch.no_grad():
                            target_layer = adapter.forward_features(target_view / 255, need_layers=True).layer_cls
                        adv_layer = adapter.forward_features(adv_view / 255, need_layers=True).layer_cls
                        coh_terms.append(fidelity_loss(adv_layer, target_layer))
                coh = torch.stack(coh_terms).mean()
                score = score - args.lambda_coh * coh
            grad = torch.autograd.grad(score, delta)[0]
            if not torch.isfinite(score) or not torch.isfinite(grad).all():
                raise FloatingPointError(f"Nonfinite score/gradient at {args.variant} image {index} step {step}")
            delta = (delta.detach() + args.alpha * grad.sign()).clamp(-args.epsilon, args.epsilon)
            if step == 0 or step + 1 == args.steps:
                last_metrics = {"foa_score": float(foa_score.detach()),
                                "extra_scores": {k: float(v.detach()) for k, v in extra_scores.items()},
                                "coh_loss": None if coh is None else float(coh.detach()),
                                "grad_l2": float(grad.norm().detach())}
        adv = (src + delta).clamp(0, 255) / 255
        save_image(adv, output_path)
        record = {"variant": args.variant, "source_id": src_path.stem,
                  "target_id": tgt_path.stem, "seed": sample_seed,
                  "runtime_sec": time.perf_counter() - start,
                  "peak_vram_bytes": torch.cuda.max_memory_allocated(device),
                  "linf_255": float(delta.abs().max()), "l2_255": float(delta.norm()),
                  "foa_previous_losses": [float(x) for x in foa_loss.previous_loss_list],
                  **last_metrics}
        with ledger.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        print("DONE", json.dumps(record), flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=VARIANTS, required=True)
    p.add_argument("--num-samples", type=int, default=5)
    p.add_argument("--seed", type=int, default=2023)
    p.add_argument("--data-root", type=Path, default=Path("/root/Hattack/datas"))
    p.add_argument("--weights-root", type=Path, default=Path("/root/.cache/huggingface"))
    p.add_argument("--out-dir", type=Path, default=Path("/root/Hattack/cache/foa_mpc_hattack"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--epsilon", type=float, default=16)
    p.add_argument("--alpha", type=float, default=1)
    p.add_argument("--lambda-coh", type=float, default=0.1)
    p.add_argument("--extra-global-weight", type=float, default=1.0)
    p.add_argument("--clip-backbones", nargs="+", choices=CLIP_NAMES,
                   default=list(CLIP_NAMES))
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.num_samples < 1 or args.steps < 1:
        raise ValueError("num-samples and steps must be positive")
    if args.variant in ("E1LDI", "E4LDI", "E4LDINoMIFOA", *COH_LDI):
        args.clip_backbones = ["Laion"]
    if args.variant in COH_LDI:
        args.lambda_coh = COH_LDI[args.variant]
    # Transformers/HF Hub may capture cache paths during module import.  A
    # single re-exec makes --weights-root effective even when passed on CLI.
    if os.environ.get("HF_HOME") != str(args.weights_root):
        env = dict(os.environ)
        env["HF_HOME"] = str(args.weights_root)
        os.execve(sys.executable, [sys.executable, "-m", "foa_mpc_hattack.run_attack", *sys.argv[1:]], env)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pairs = find_pairs(args.data_root, args.num_samples)
    with (args.out_dir / "pair_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("index", "source_id", "target_id", "source_path", "target_path"))
        writer.writerows((i, s.stem, t.stem, str(s), str(t)) for i, (s, t) in enumerate(pairs))
    with (args.out_dir / f"{args.variant}_config.json").open("w", encoding="utf-8") as stream:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, stream, indent=2)
    import transformers
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
    metadata = {"torch": torch.__version__, "transformers": transformers.__version__,
                "git_commit": git_commit,
                "foa_commit": "6aa9ad463a1e812b7f77ff2e5c4b1712e7c78c4b",
                "mpc_commit": "7b195a5bcbaeaf1e8534d101199810a2782caad3",
                "models": ["openai/clip-vit-base-patch16", "openai/clip-vit-base-patch32",
                           "laion/CLIP-ViT-G-14-laion2B-s12B-b42K",
                           "facebook/dinov2-base", "OpenGVLab/InternVL3-1B-hf"],
                "losses": {"CLIP_B16": {"foa_ot": "B16" in args.clip_backbones and args.variant not in ("E4", "E4NoMI", "E4NoMIFOA", "E4LDI"), "coherence": args.variant in ("E2", "E3")},
                           "CLIP_B32": {"foa_ot": "B32" in args.clip_backbones and args.variant not in ("E4", "E4NoMI", "E4NoMIFOA", "E4LDI"), "coherence": args.variant in ("E2", "E3")},
                           "CLIP_LAION": {"foa_ot": "Laion" in args.clip_backbones and args.variant not in ("E4", "E4NoMI", "E4NoMIFOA", "E4LDI"), "coherence": args.variant in ("E2", "E3", *COH_LDI)},
                           "DINOv2": {"global_cosine": args.variant in ("E1", "E1LDI", "E2", "E3", *COH_LDI), "coherence": args.variant == "E3" or args.variant in COH_LDI, "local_ot_enabled": False},
                           "InternVL3_1B": {"global_cosine": args.variant in ("E1", "E1LDI", "E2", "E3", *COH_LDI), "coherence": args.variant == "E3" or args.variant in COH_LDI, "local_ot_enabled": False},
                           "E4": {"mpc_original_infonce": args.variant in ("E4", "E4NoMI", "E4NoMIFOA", "E4LDI")}}}
    (args.out_dir / "environment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if args.variant in ("E4", "E4NoMI", "E4NoMIFOA", "E4LDI", "E4LDINoMIFOA"):
        from .run_mpc_original import run_original
        run_original(args, pairs)
    else:
        run_foa_family(args, pairs, args.out_dir)


if __name__ == "__main__":
    main()
