import torch
from torchvision import transforms

from config import Config
from extractors import BACKBONE_MAP

def init_backbones(cfg: Config):
    backbones = []
    for key in cfg.model.backbones:
        backbones.append(
            BACKBONE_MAP[key]().eval().to(cfg.device).requires_grad_(False)
        )
    return backbones

class BaseTrainer:
    extractor = None
    loss = None

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.backbones = init_backbones(cfg)
        self.source_crop = (
            transforms.RandomResizedCrop(cfg.model.resize, scale=cfg.model.crop_scale)
            if cfg.model.use_source_crop
            else torch.nn.Identity()
        )
        self.target_crop = (
            transforms.RandomResizedCrop(cfg.model.resize, scale=cfg.model.crop_scale)
            if cfg.model.use_target_crop
            else torch.nn.Identity()
        )
    
    def attack_image(self, src, tgt):
        pass