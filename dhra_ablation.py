"""P0 DHRA ablations using the existing feature extractor and optimizer settings."""
import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from config import Config, TomlLoader
from methods.dhra import DHRATrainer
from methods.dhra_loss import coherence, density_state, psd_sqrt
from utils import set_seed, to_tensor

VARIANTS = {
    'A0': ('none', False, False),
    'A1': ('quantum', False, False),
    'A2': ('none', True, True),
    'A3': ('quantum', True, True),
    'B1': ('frobenius', False, False),
    'B2': ('gram_cosine', False, False),
    'B3': ('layer_cosine', False, False),
    'C1': ('quantum', True, False),
    'C2': ('quantum', False, True),
}


def fidelity_stable(target_state, adversarial_state):
    """Same squared fidelity, using eigenvalues only for the inner trace.

    The original reconstructs sqrt(middle) from eigenvectors before taking its
    trace. Eigenvector derivatives are undefined for repeated eigenvalues;
    tr(sqrt(middle)) equals the sum of sqrt(eigenvalues) exactly.
    """
    root = psd_sqrt(target_state)
    middle = root @ adversarial_state @ root
    middle = (middle + middle.transpose(-1, -2)) * 0.5
    eigenvalues = torch.linalg.eigvalsh(middle)
    return eigenvalues.clamp_min(1e-8).sqrt().sum(-1).square().clamp(max=1.0)


def components(adv_features, target_features, alignment):
    bases, aligns, decs = [], [], []
    for adv, target in zip(adv_features, target_features):
        qa, qt = density_state(adv), density_state(target)
        bases.append(1 - F.cosine_similarity(adv[:, -1], target[:, -1], dim=-1))
        if alignment == 'quantum':
            aligns.append(1 - fidelity_stable(qt, qa))
        elif alignment == 'frobenius':
            aligns.append((qa - qt).square().sum(dim=(-2, -1)) / qa.shape[-1] ** 2)
        elif alignment == 'gram_cosine':
            aligns.append(1 - F.cosine_similarity(qa.flatten(1), qt.flatten(1), dim=-1))
        elif alignment == 'layer_cosine':
            aligns.append(1 - F.cosine_similarity(adv, target, dim=-1).mean(-1))
        elif alignment == 'none':
            aligns.append(torch.zeros_like(bases[-1]))
        else:
            raise ValueError(alignment)
        target_c, adv_c = coherence(qt), coherence(qa)
        decs.append((target_c - adv_c).clamp_min(0) / (target_c + 1e-8))
    decs = torch.stack(decs)
    return {'base': torch.stack(bases).mean(),
            'align': torch.stack(aligns).mean(),
            'dec_mean': decs.mean(),
            'dec_variance': decs.var(dim=0, unbiased=False).mean()}


def total(parts, alignment, mean_enabled, variance_enabled, cfg):
    return (parts['base']
            + (cfg.optim.lambda_coh * parts['align'] if alignment != 'none' else 0)
            + (cfg.optim.lambda_dec * parts['dec_mean'] if mean_enabled else 0)
            + (cfg.optim.lambda_dec * cfg.optim.gamma * parts['dec_variance'] if variance_enabled else 0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', choices=VARIANTS, required=True)
    p.add_argument('--limit', type=int, default=1000)
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--seed', type=int, default=2023)
    p.add_argument('--log-every', type=int, default=30)
    args = p.parse_args()
    if not 1 <= args.limit <= 1000 or args.steps < 1:
        p.error('limit must be 1..1000 and steps positive')
    cfg = TomlLoader('config.toml', Config)
    if cfg.data.src_data_path != 'datas/nips2017_1000' or cfg.data.tgt_data_path != 'datas/coco_val2017_1000':
        raise ValueError('Main source/target pairing differs from NIPS/COCO configuration')
    method = 'DHRA_ABL_' + args.variant
    alignment, mean_enabled, variance_enabled = VARIANTS[args.variant]
    output = Path('cache/adv_images') / method
    logs = Path('cache/ablation') / method
    output.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    metadata = {'variant': args.variant, 'method': method, 'alignment': alignment,
                'decoherence_mean': mean_enabled, 'decoherence_variance': variance_enabled,
                'samples': args.limit, 'steps': args.steps, 'seed': args.seed,
                'epsilon': cfg.optim.epsilon, 'alpha': cfg.optim.alpha,
                'lambda_coh': cfg.optim.lambda_coh, 'lambda_dec': cfg.optim.lambda_dec,
                'gamma': cfg.optim.gamma, 'backbones': cfg.model.backbones,
                'crop_scale': cfg.model.crop_scale, 'source_crop': cfg.model.use_source_crop,
                'target_crop': cfg.model.use_target_crop}
    (logs / 'config.json').write_text(json.dumps(metadata, indent=2))
    set_seed(args.seed)
    trainer = DHRATrainer(cfg)
    pre = transforms.Compose([transforms.Resize(cfg.model.resize, interpolation=transforms.InterpolationMode.BICUBIC),
                              transforms.CenterCrop(cfg.model.resize)])
    def load(path):
        with Image.open(path) as im:
            return to_tensor(pre(im.convert('RGB'))).unsqueeze(0).to(cfg.device)

    for i in range(args.limit):
        name = f'{i:04d}.png'
        image_path = output / name
        record_path = logs / f'{i:04d}.json'
        if image_path.is_file() and record_path.is_file():
            previous = json.loads(record_path.read_text())
            if previous.get('steps') == args.steps and previous.get('seed') == args.seed + i:
                print('SKIP', args.variant, name, flush=True)
                continue
        set_seed(args.seed + i)
        source = load(Path(cfg.data.src_data_path) / name)
        target = load(Path(cfg.data.tgt_data_path) / name)
        delta = torch.zeros_like(source)
        history = []
        start = perf_counter()
        for step in range(args.steps):
            delta = delta.detach().requires_grad_(True)
            with torch.no_grad():
                tf = trainer.features(trainer.target_crop(target))
            af = trainer.features(trainer.source_crop((source + delta).clamp(0, 255)))
            parts = components(af, tf, alignment)
            loss = total(parts, alignment, mean_enabled, variance_enabled, cfg)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'{args.variant} {name} step {step}: nonfinite loss')
            grad = torch.autograd.grad(loss, delta, retain_graph=(step == 0))[0]
            if not torch.isfinite(grad).all():
                raise FloatingPointError(f'{args.variant} {name} step {step}: nonfinite gradient')
            if step == 0 or (step + 1) % args.log_every == 0 or step == args.steps - 1:
                row = {'step': step + 1, 'total': float(loss.detach()),
                       **{key: float(value.detach()) for key, value in parts.items()},
                       'total_grad_l2': float(grad.norm().detach())}
                if step == 0:
                    row['component_grad_l2'] = {}
                    for key, value in parts.items():
                        if value.requires_grad:
                            g = torch.autograd.grad(value, delta, retain_graph=True, allow_unused=True)[0]
                            row['component_grad_l2'][key] = float(g.norm().detach()) if g is not None else 0.0
                history.append(row)
            delta = (delta - cfg.optim.alpha * grad.sign()).clamp(-cfg.optim.epsilon, cfg.optim.epsilon)
            delta = ((source + delta).clamp(0, 255) - source).detach()
        save_image((source + delta).clamp(0, 255) / 255, image_path)
        torch.save(delta.cpu().to(torch.int8), logs / f'{i:04d}_delta.pt')
        result = {'image': name, 'steps': args.steps, 'seed': args.seed + i,
                  'seconds': round(perf_counter() - start, 2),
                  'linf': float(delta.abs().max()), 'history': history}
        record_path.write_text(json.dumps(result, indent=2))
        print('DONE', args.variant, name, result['seconds'], 'linf', result['linf'], flush=True)
    print('COMPLETE', args.variant, args.limit, flush=True)


if __name__ == '__main__':
    main()
