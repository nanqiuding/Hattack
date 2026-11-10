"""DHRA attack using all visual Transformer blocks of each CLIP surrogate."""
from pathlib import Path
from time import perf_counter
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from .base_trainer import BaseTrainer
from .dhra_loss import dhra_loss


class DHRATrainer(BaseTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.save_path = Path('cache/adv_images') / cfg.method
        self.save_path.mkdir(parents=True, exist_ok=True)

    def features(self, image):
        result = []
        for backbone in self.backbones:
            pixels = backbone.normalizer(image)
            states = backbone.model.vision_model(
                pixel_values=pixels, output_hidden_states=True
            ).hidden_states
            # hidden_states[0] is the pre-Transformer embedding, not a block.
            result.append(torch.stack([s[:, 0, :] for s in states[1:]], dim=1))
        return result

    def attack_image(self, src, tgt):
        outputs = [self.save_path / f'{Path(path).stem}.png' for path in src['path']]
        if all(path.exists() for path in outputs):
            print(f'SKIP existing {outputs[0]}', flush=True)
            return
        start = perf_counter()
        print(f'START {outputs[0].name}', flush=True)
        source = src['img'].to(self.cfg.device)
        target = tgt['img'].to(self.cfg.device)
        delta = torch.zeros_like(source)
        epsilon = self.cfg.optim.epsilon
        alpha = self.cfg.optim.alpha
        for _ in range(self.cfg.optim.steps):
            delta = delta.detach().requires_grad_(True)
            with torch.no_grad():
                target_features = self.features(self.target_crop(target))
            adversarial_features = self.features(self.source_crop((source + delta).clamp(0, 255)))
            loss = dhra_loss(adversarial_features, target_features,
                             self.cfg.optim.lambda_coh, self.cfg.optim.lambda_dec,
                             self.cfg.optim.gamma)
            gradient = torch.autograd.grad(loss, delta)[0]
            delta = (delta - alpha * gradient.sign()).clamp(-epsilon, epsilon)
            delta = (source + delta).clamp(0, 255) - source
        for image, path in zip((source + delta).clamp(0, 255) / 255, src['path']):
            save_image(image, self.save_path / f'{Path(path).stem}.png')
        print(f'DONE {outputs[0].name} seconds={perf_counter()-start:.1f}', flush=True)
