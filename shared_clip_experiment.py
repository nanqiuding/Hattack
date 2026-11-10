"""Fit B16-anchored CLIP maps and run controlled targeted attacks."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from torchvision import transforms
from torchvision.utils import save_image

from extractors import BACKBONE_MAP
from utils import to_tensor, set_seed

ROOT = Path('cache/analysis/shared_clip_b16')
SRC = Path('datas/nips2017_1000')
TGT = Path('datas/coco_val2017_1000')
VARIANTS = {
    'single_b16': 'SHARED_B16_NIPS100',
    'aligned_b16_b32': 'SHARED_B16_B32_NIPS100',
    'aligned_pca': 'SHARED_PCA_NIPS100',
    'aligned_learned': 'SHARED_LEARNED_NIPS100',
}


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def extract(name, paths):
    model = BACKBONE_MAP[name]().eval().cuda().requires_grad_(False)
    rows = []
    with torch.inference_mode():
        for i, path in enumerate(paths, 1):
            with Image.open(path) as opened:
                image = transforms.CenterCrop(224)(transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC)(opened.convert('RGB')))
            x = to_tensor(image).unsqueeze(0).cuda()
            rows.append(model.global_features(x)[0].squeeze(0).cpu().numpy())
            if i % 100 == 0:
                print('EXTRACT', name, i, flush=True)
    del model
    torch.cuda.empty_cache()
    return np.stack(rows).astype(np.float64)


def procrustes(x, y):
    rotation, trace = orthogonal_procrustes(x, y)
    scale = float(trace / np.sum(x * x))
    return rotation, scale


def align(x, mx, projection, rotation, scale, my):
    return scale * ((x - mx) @ projection) @ rotation + my


def metrics(a, b):
    a, b = unit(a), unit(b)
    sim = a @ b.T
    return {'cosine_mean': float(np.diag(sim).mean()),
            'top1_retrieval': float(np.mean(np.argmax(sim, axis=1) == np.arange(len(a))))}


def fit():
    ROOT.mkdir(parents=True, exist_ok=True)
    ids = list(range(100, 1000))
    paths = [SRC / f'{i:04d}.png' for i in ids]
    if any(not p.is_file() for p in paths):
        raise FileNotFoundError('NIPS alignment images missing')
    features = {name: extract(name, paths) for name in ('B16', 'B32', 'Laion')}
    train = np.arange(0, 800)  # IDs 0100..0899
    val = np.arange(800, 900)  # IDs 0900..0999
    b, c, g = (features[name] for name in ('B16', 'B32', 'Laion'))
    mb, mc, mg = (a[train].mean(axis=0) for a in (b, c, g))
    rb, sb = procrustes(c[train] - mc, b[train] - mb)
    identity = np.eye(768)
    pca = PCA(n_components=768, svd_solver='randomized', random_state=2026)
    pca.fit(g[train] - mg)
    pg = pca.components_.T
    rp, sp = procrustes((g[train] - mg) @ pg, b[train] - mb)
    # Supervised linear dimension reduction; ridge coefficients are learned from
    # paired training features. Select regularization using held-out IDs only.
    candidates = []
    for alpha in (0.01, 0.1, 1.0, 10.0):
        ridge = Ridge(alpha=alpha, fit_intercept=False).fit(g[train] - mg, b[train] - mb)
        pl = ridge.coef_.T
        rl, sl = procrustes((g[train] - mg) @ pl, b[train] - mb)
        score = metrics(align(g[val], mg, pl, rl, sl, mb), b[val])['cosine_mean']
        candidates.append((score, alpha, pl, rl, sl))
        print('RIDGE_VALIDATION', alpha, score, flush=True)
    score, alpha, pl, rl, sl = max(candidates, key=lambda a: a[0])
    np.savez(ROOT / 'maps.npz', mean_b16=mb, mean_b32=mc, mean_g=mg,
             rotation_b32=rb, scale_b32=sb, projection_pca=pg,
             rotation_pca=rp, scale_pca=sp, projection_learned=pl,
             rotation_learned=rl, scale_learned=sl)
    result = {
        'anchor': 'B16 final CLS before projection',
        'train_ids': '0100..0899', 'validation_ids': '0900..0999',
        'attack_ids': '0000..0099', 'learned_projection': 'supervised ridge',
        'ridge_alpha_selected': alpha,
        'validation': {
            'B32_unaligned': metrics(c[val], b[val]),
            'B32_aligned': metrics(align(c[val], mc, identity, rb, sb, mb), b[val]),
            'G_pca_aligned': metrics(align(g[val], mg, pg, rp, sp, mb), b[val]),
            'G_learned_aligned': metrics(align(g[val], mg, pl, rl, sl, mb), b[val]),
        },
    }
    (ROOT / 'fit_summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


class FusedModel:
    def __init__(self, variant):
        self.variant = variant
        self.names = ['B16']
        if variant != 'single_b16': self.names.append('B32')
        if variant in ('aligned_pca', 'aligned_learned'): self.names.append('Laion')
        self.models = {name: BACKBONE_MAP[name]().eval().cuda().requires_grad_(False)
                       for name in self.names}
        if variant != 'single_b16':
            values = np.load(ROOT / 'maps.npz')
            self.maps = {key: torch.tensor(values[key], device='cuda', dtype=torch.float32)
                         for key in values.files}

    def __call__(self, image):
        vectors = [self.models['B16'].global_features(image)[0]]
        if 'B32' in self.models:
            h = self.models['B32'].global_features(image)[0]
            m = self.maps
            aligned = m['scale_b32'] * ((h - m['mean_b32']) @ m['rotation_b32']) + m['mean_b16']
            vectors.append(F.normalize(aligned, dim=1))
        if 'Laion' in self.models:
            h = self.models['Laion'].global_features(image)[0]
            suffix = 'pca' if self.variant == 'aligned_pca' else 'learned'
            m = self.maps
            aligned = m[f'scale_{suffix}'] * (((h - m['mean_g']) @ m[f'projection_{suffix}'])
                                                 @ m[f'rotation_{suffix}']) + m['mean_b16']
            vectors.append(F.normalize(aligned, dim=1))
        return F.normalize(torch.stack(vectors).mean(dim=0), dim=1)


def attack(variant):
    set_seed()
    model = FusedModel(variant)
    crop = transforms.RandomResizedCrop(224, scale=(0.5, 0.9))
    output = Path('cache/adv_images') / VARIANTS[variant]
    output.mkdir(parents=True, exist_ok=True)
    for i in range(100):
        dest = output / f'{i:04d}.png'
        if dest.exists():
            print('SKIP', dest.name, flush=True)
            continue
        with Image.open(SRC / f'{i:04d}.png') as im:
            src = to_tensor(transforms.CenterCrop(224)(transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC)(im.convert('RGB')))).unsqueeze(0).cuda()
        with Image.open(TGT / f'{i:04d}.png') as im:
            tgt = to_tensor(transforms.CenterCrop(224)(transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC)(im.convert('RGB')))).unsqueeze(0).cuda()
        start = perf_counter()
        delta = torch.zeros_like(src)
        for _ in range(300):
            delta = delta.detach().requires_grad_(True)
            adv_feature = model(crop((src + delta).clamp(0, 255)))
            with torch.no_grad():
                tgt_feature = model(crop(tgt))
            loss = -F.cosine_similarity(adv_feature, tgt_feature, dim=1).mean()
            grad = torch.autograd.grad(loss, delta)[0]
            delta = (delta - grad.sign()).clamp(-16, 16)
            delta = (src + delta).clamp(0, 255) - src
        save_image((src + delta).clamp(0, 255) / 255, dest)
        print('DONE', variant, dest.name, round(perf_counter() - start, 1), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('fit', 'attack'))
    parser.add_argument('--variant', choices=VARIANTS)
    args = parser.parse_args()
    if args.action == 'fit': fit()
    else:
        if not args.variant: parser.error('--variant required for attack')
        attack(args.variant)
