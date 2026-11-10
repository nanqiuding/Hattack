"""Test whether an orthogonal map aligns B/16 and B/32 image CLS features."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.linalg import orthogonal_procrustes
from sklearn.linear_model import Ridge
from extractors import BACKBONE_MAP
from utils import to_tensor


def normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def extract(name, paths):
    model = BACKBONE_MAP[name]().eval().cuda().requires_grad_(False)
    out = []
    with torch.inference_mode():
        for i, path in enumerate(paths):
            image = Image.open(path).convert('RGB').resize((224, 224))
            x = to_tensor(image).unsqueeze(0).cuda()
            out.append(model.global_features(x)[0].squeeze(0).cpu().numpy())
            if (i + 1) % 100 == 0:
                print(name, i + 1, flush=True)
    del model
    torch.cuda.empty_cache()
    return normalize(np.stack(out).astype(np.float64))


def scores(pred, target):
    pred, target = normalize(pred), normalize(target)
    sim = pred @ target.T
    ranks = np.argsort(-sim, axis=1)
    return {
        'same_image_cosine_mean': float(np.diag(sim).mean()),
        'same_image_cosine_median': float(np.median(np.diag(sim))),
        'top1_retrieval': float(np.mean(ranks[:, 0] == np.arange(len(pred)))),
        'top5_retrieval': float(np.mean(np.any(ranks[:, :5] == np.arange(len(pred))[:, None], axis=1))),
        'matched_minus_other_cosine': float(np.diag(sim).mean() - (sim.sum() - np.trace(sim)) / (sim.size - len(sim))),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--images', type=Path, default=Path('datas/nips2017_1000'))
    p.add_argument('--output', type=Path, default=Path('cache/analysis/b16_b32_procrustes_1000.json'))
    p.add_argument('--limit', type=int, default=1000)
    args = p.parse_args()
    paths = sorted(args.images.glob('*.png'))[:args.limit]
    assert len(paths) == args.limit, len(paths)
    x, y = extract('B16', paths), extract('B32', paths)
    assert x.shape == y.shape == (args.limit, 768), (x.shape, y.shape)
    rng = np.random.default_rng(2026)
    permutation = rng.permutation(args.limit)
    split = int(args.limit * 0.8)
    train, test = permutation[:split], permutation[split:]
    mx, my = x[train].mean(axis=0), y[train].mean(axis=0)
    xc, yc = x[train] - mx, y[train] - my
    rotation, scale_sum = orthogonal_procrustes(xc, yc)
    scale = float(scale_sum / np.sum(xc * xc))
    # Evaluate both a pure rotation and a similarity transform. The translation
    # is entirely determined by the training-set means.
    methods = {
        'identity': (x[test], y[test]),
        'rotation_only': (x[test] @ rotation, y[test]),
        'rotation_centered': ((x[test] - mx) @ rotation + my, y[test]),
        'rotation_scale_centered': (scale * (x[test] - mx) @ rotation + my, y[test]),
    }
    # A flexible baseline makes it clear whether the orthogonal constraint costs accuracy.
    ridge = Ridge(alpha=1e-2, fit_intercept=True).fit(x[train], y[train])
    methods['ridge_linear'] = (ridge.predict(x[test]), y[test])
    result = {
        'representation': 'L2-normalized final-layer CLS before CLIP projection',
        'source': 'official NIPS2017 images',
        'image_count': args.limit,
        'train_count': len(train),
        'test_count': len(test),
        'random_seed': 2026,
        'feature_dimension': 768,
        'procrustes_scale': scale,
        'test_image_ids': [paths[i].stem for i in test],
        'test_metrics': {name: scores(*pair) for name, pair in methods.items()},
        'train_metrics': {
            'identity': scores(x[train], y[train]),
            'rotation_centered': scores(xc @ rotation + my, y[train]),
            'rotation_scale_centered': scores(scale * xc @ rotation + my, y[train]),
            'ridge_linear': scores(ridge.predict(x[train]), y[train]),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result['test_metrics'], indent=2), flush=True)
    print('OUTPUT', args.output, flush=True)


if __name__ == '__main__':
    main()
