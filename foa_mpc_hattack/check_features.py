"""One-model-at-a-time shape and pixel-gradient check with real checkpoints."""
import argparse
import json
import os
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True,
                   choices=("CLIP_B16", "CLIP_B32", "CLIP_LAION", "DINOv2", "InternVL3_1B"))
    p.add_argument("--weights-root", default="/root/.cache/huggingface")
    args = p.parse_args()
    if os.environ.get("HF_HOME") != args.weights_root:
        env = dict(os.environ)
        env["HF_HOME"] = args.weights_root
        os.execve(sys.executable, [sys.executable, "-m", "foa_mpc_hattack.check_features", *sys.argv[1:]], env)
    import torch
    from .adapters import FOAClipAdapter, DINOv2Adapter, InternVL3Adapter
    device = torch.device("cuda:0")
    adapter = (FOAClipAdapter(args.model, device) if args.model.startswith("CLIP_") else
               DINOv2Adapter(device) if args.model == "DINOv2" else InternVL3Adapter(device))
    image = torch.rand(1, 3, 224, 224, device=device, requires_grad=True)
    output = adapter.forward_features(image, need_patches=True, need_layers=args.model.startswith("CLIP_"))
    value = output.global_feat.square().sum() + 0.1 * output.global_feat.sum()
    grad, = torch.autograd.grad(value, image)
    if not torch.isfinite(grad).all() or float(grad.norm()) == 0:
        raise AssertionError(f"No finite pixel gradient for {args.model}")
    report = {"model": args.model, "global_shape": list(output.global_feat.shape),
              "patch_shape": None if output.patch_feats is None else list(output.patch_feats.shape),
              "layer_cls_shape": None if output.layer_cls is None else list(output.layer_cls.shape),
              "pixel_grad_l2": float(grad.norm()),
              "peak_vram_bytes": torch.cuda.max_memory_allocated(device)}
    print("FEATURES_OK", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
