"""AdvDiffVLM-style SD1.4 adaptation; not the paper's class-conditional model."""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from transformers import CLIPTextModel, CLIPTokenizer

from extractors import BACKBONE_MAP
from utils import to_tensor


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--limit', type=int, default=100)
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--rounds', type=int, default=2)
    p.add_argument('--strength', type=float, default=15.0)
    p.add_argument('--max-t', type=int, default=200)
    p.add_argument('--method', default='ADVDIFFVLM_ADAPTED_NIPS100')
    args = p.parse_args()
    torch.manual_seed(2026)
    device = 'cuda'
    root = '/root/autodl-tmp/sd_v1_4'
    unet = UNet2DConditionModel.from_pretrained(root, subfolder='unet', torch_dtype=torch.float16, local_files_only=True).to(device).eval().requires_grad_(False)
    vae = AutoencoderKL.from_pretrained(root, subfolder='vae', torch_dtype=torch.float16, local_files_only=True).to(device).eval().requires_grad_(False)
    text = CLIPTextModel.from_pretrained(root, subfolder='text_encoder', torch_dtype=torch.float16, local_files_only=True).to(device).eval().requires_grad_(False)
    tok = CLIPTokenizer.from_pretrained(root, subfolder='tokenizer', local_files_only=True)
    sched = DDIMScheduler.from_pretrained(root, subfolder='scheduler', local_files_only=True)
    sched.set_timesteps(args.steps)
    ts = [int(t) for t in sched.timesteps if int(t) <= args.max_t]
    if len(ts) < 3:
        raise ValueError('Too few reverse timesteps; raise --steps or --max-t')
    with torch.no_grad():
        ids = tok([''], padding='max_length', max_length=tok.model_max_length, return_tensors='pt').input_ids.to(device)
        context = text(ids)[0]
    del text
    models = {name: BACKBONE_MAP[name]().eval().to(device).requires_grad_(False) for name in ('B16', 'B32', 'Laion')}
    pre = transforms.Compose([transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC), transforms.CenterCrop(224)])

    def load(path):
        with Image.open(path) as im:
            return (to_tensor(pre(im.convert('RGB'))) / 255).unsqueeze(0).to(device)

    def enc(x):
        with torch.no_grad():
            return vae.encode((x * 2 - 1).half()).latent_dist.mode() * vae.config.scaling_factor

    def dec(z):
        return ((vae.decode(z / vae.config.scaling_factor).sample.float() + 1) / 2).clamp(0, 1)

    def feat(name, x):
        return F.normalize(models[name].global_features(x * 255)[0].float(), dim=1)

    def saliency(source, target_feature):
        x = source.detach().requires_grad_(True)
        score = (feat('B16', x) * target_feature).sum()
        grad = torch.autograd.grad(score, x)[0].abs().mean(1, keepdim=True)
        grad = F.avg_pool2d(grad, 15, stride=1, padding=7)
        grad = F.interpolate(grad, size=(28, 28), mode='bilinear', align_corners=False)
        return (grad / grad.amax().clamp_min(1e-8)).detach()

    out = Path('cache/adv_images') / args.method
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(args.limit):
        name = f'{i:04d}.png'
        path = out / name
        if path.is_file():
            print('SKIP', name, flush=True)
            continue
        start = time.monotonic()
        source = load(Path('datas/nips2017_1000') / name)
        target = load(Path('datas/coco_val2017_1000') / name)
        target_features = {key: feat(key, target).detach() for key in models}
        cam = saliency(source, target_features['B16'])
        base = enc(source)
        last_scores = None
        for round_index in range(args.rounds):
            noise = torch.randn_like(base)
            first = ts[0]
            alpha = sched.alphas_cumprod[first].to(device)
            z = alpha.sqrt() * base + (1 - alpha).sqrt() * noise
            score_history = []
            for k, t in enumerate(ts):
                a = sched.alphas_cumprod[t].to(device)
                original_t = a.sqrt() * base + (1 - a).sqrt() * noise
                mask = cam.clamp(0.045, 1)
                # Retain a random salient patch of the evolving latent.
                patch = torch.multinomial(cam.flatten(), 1).item()
                cy, cx = divmod(patch, cam.shape[-1])
                mask = mask.clone()
                mask[..., max(0, cy-2):cy+2, max(0, cx-2):cx+2] = 0
                z = (mask * original_t + (1 - mask) * z).half()
                with torch.no_grad():
                    eps = unet(z, torch.tensor(t, device=device), encoder_hidden_states=context).sample
                    z = sched.step(eps, t, z, eta=0.0).prev_sample.detach().half()
                if k < int(0.8 * len(ts)):
                    continue
                if len(score_history) < 2:
                    weights = torch.ones(len(models), device=device)
                else:
                    ratio = score_history[-1] / score_history[-2].clamp_min(1e-5)
                    weights = torch.softmax(ratio / 2, dim=0) * len(models)
                z = z.detach().requires_grad_(True)
                image = dec(z)
                scores = torch.stack([(feat(key, image) * target_features[key]).sum() for key in models])
                objective = (scores * weights).sum() / len(models)
                grad = torch.autograd.grad(objective, z)[0]
                z = (z.detach() + args.strength * grad.clamp(-0.0025, 0.0025)).detach()
                score_history.append(scores.detach())
            with torch.no_grad():
                image = dec(z).detach()
                base = enc(image)
            last_scores = scores.detach().cpu().tolist() if score_history else None
        save_image(image, path)
        row = {'id': name, 'clip_cosines': last_scores, 'seconds': round(time.monotonic()-start, 2),
               'linf': float((image-source).abs().max())}
        rows.append(row)
        print('RESULT', json.dumps(row), flush=True)
        (out / 'run_metadata.json').write_text(json.dumps({'adaptation': 'SD1.4, B16/B32/LAION, CLIP-gradient saliency; not official AdvDiffVLM', 'config': vars(args), 'rows': rows}, indent=2))
    print('COMPLETE', len(list(out.glob('*.png'))), flush=True)


if __name__ == '__main__':
    main()
