"""M-Attack baseline: ensemble CLIP CLS cosine alignment with I-FGSM."""
from pathlib import Path
from time import perf_counter
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from .base_trainer import BaseTrainer

class MAttackTrainer(BaseTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.save_path = Path('cache/adv_images') / cfg.method
        self.save_path.mkdir(parents=True, exist_ok=True)

    def attack_image(self, src, tgt):
        outputs = [self.save_path / f'{Path(path).stem}.png' for path in src['path']]
        if all(path.exists() for path in outputs):
            print('SKIP', outputs[0], flush=True)
            return
        start = perf_counter()
        source = src['img'].to(self.cfg.device)
        target = tgt['img'].to(self.cfg.device)
        delta = torch.zeros_like(source)
        epsilon = self.cfg.optim.epsilon
        alpha = self.cfg.optim.alpha
        for _ in range(self.cfg.optim.steps):
            delta = delta.detach().requires_grad_(True)
            adv_crop = self.source_crop((source + delta).clamp(0, 255))
            tgt_crop = self.target_crop(target)
            loss = 0
            for backbone in self.backbones:
                z_adv = backbone.global_features(adv_crop)[0]
                with torch.no_grad():
                    z_tgt = backbone.global_features(tgt_crop)[0]
                loss = loss - F.cosine_similarity(z_adv, z_tgt, dim=1).mean()
            loss = loss / len(self.backbones)
            grad = torch.autograd.grad(loss, delta)[0]
            delta = (delta - alpha * grad.sign()).clamp(-epsilon, epsilon)
            delta = (source + delta).clamp(0, 255) - source
        for image, output in zip((source + delta).clamp(0, 255) / 255, outputs):
            save_image(image, output)
        print('DONE', outputs[0].name, f'seconds={perf_counter()-start:.1f}', flush=True)
