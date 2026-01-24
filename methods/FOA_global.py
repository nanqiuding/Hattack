from pathlib import Path
from torchvision.utils import save_image

from config import Config
from extractors import (
    EnsembleExtractor_global,
    EnsembleLoss_global
)

from .base_trainer import BaseTrainer
from .attack_function import ATTACK_FUNC_MAP

class FOA_global_trainer(BaseTrainer):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.extractor = EnsembleExtractor_global(self.backbones)
        self.loss = EnsembleLoss_global(self.backbones)

        self.save_path = Path("cache/adv_images") / self.cfg.method
        self.save_path.mkdir(parents=True, exist_ok=True)

    def attack_image(self, src, tgt):
        print(f"attack images: {src['path']} to {tgt['path']}")

        adv_image = ATTACK_FUNC_MAP[self.cfg.attack](
            optim=self.cfg.optim,
            ensemble_extractor=self.extractor,
            ensemble_loss=self.loss,
            source_crop=self.source_crop,
            target_crop=self.target_crop,
            src_img=src["img"].to(self.cfg.device),
            tgt_img=tgt["img"].to(self.cfg.device),
        )

        for adv, path in zip(adv_image, src["path"]):
            save_image(adv, self.save_path / f"{Path(path).stem}.png")