import torch
from pathlib import Path
from torchvision.utils import save_image

from config import Config
from extractors import (
    EnsembleExtractor_random,
    EnsembleLoss_random
)

from .base_trainer import BaseTrainer
from .attack_function import ATTACK_FUNC_MAP

class FOA_inter_30_intra_trainer(BaseTrainer):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.extractor = EnsembleExtractor_random(self.backbones)
        self.loss = EnsembleLoss_random(self.backbones)
    
    def attack_image(self, src, tgt):
        print(f"attack images: {src['path']} to {tgt['path']}")
        for cluster_num in (3, 5):
            save_path = Path("cache/adv_images") / self.cfg.method / f"cluster_{cluster_num}"
            save_path.mkdir(parents=True, exist_ok=True)

            adv_image = self.attack_cluster(src["img"], tgt["img"], cluster_num)
            for adv, path in zip(adv_image, src["path"]):
                save_image(adv, save_path / f"{Path(path).stem}.png")
    
    def attack_cluster(self, src_img: torch.Tensor, tgt_img: torch.Tensor, cluster_num: int):
        src_img = src_img.to(self.cfg.device)
        tgt_img = tgt_img.to(self.cfg.device)

        self.extractor.cluster_number = cluster_num
        self.loss.cluster_number = cluster_num

        return ATTACK_FUNC_MAP[self.cfg.attack](
            optim=self.cfg.optim,
            ensemble_extractor=self.extractor,
            ensemble_loss=self.loss,
            source_crop=self.source_crop,
            target_crop=self.target_crop,
            src_img=src_img,
            tgt_img=tgt_img,
        )