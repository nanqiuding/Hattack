import torch
from torch import Tensor

from .function import OT, ComplexLoss
from .clip_extractor import ClipExtractor
from .ensemble_extractor import EnsembleExtractor, EnsembleLoss

class EnsembleExtractor_global(EnsembleExtractor):
    def __init__(self, extractors: list[ClipExtractor]):
        super().__init__(extractors)

    def forward(self, x: Tensor):
        global_features = []

        for model in self.extractors:
            global_f = model.global_features(x)
            global_features.append(global_f.squeeze())

        return global_features,

class EnsembleLoss_global(EnsembleLoss):
    def __init__(self, extractors: list[ClipExtractor]):
        super().__init__(extractors)
        self.global_features_gt = []
        self.prev_loss = []

    @torch.no_grad()
    def set_ground_truth(self, x: Tensor):
        self.global_features_gt.clear()

        for model in self.extractors:
            global_f = model.global_features(x)
            self.global_features_gt.append(global_f)

    def forward(self, global_features: list[Tensor]):
        global_loss = []

        for index in range(len(self.extractors)):
            global_gt = self.global_features_gt[index]
            global_f = global_features[index].unsqueeze(0)
            g_loss = OT(global_gt, global_f)
            global_loss.append(g_loss)

        return ComplexLoss(global_loss, self.prev_loss)