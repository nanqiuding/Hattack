import torch
from torch import Tensor

from .clip_extractor import ClipExtractor
from .function import OT, ComplexLoss, get_cluster_center
from .ensemble_extractor import EnsembleExtractor, EnsembleLoss

class EnsembleExtractor_OT_FOA(EnsembleExtractor):
    def __init__(self, extractors: list[ClipExtractor], cluster_number=5):
        super().__init__(extractors)
        self.cluster_number = cluster_number

    def forward(self, x: Tensor):
        global_features = []
        local_features = []

        for model in self.extractors:
            global_f, local_f = model.global_local_features(x)
            global_features.append(global_f.squeeze())

            local_f_center = get_cluster_center(local_f[0], self.cluster_number).unsqueeze(0)
            local_features.append(local_f_center)

        return global_features, local_features

class EnsembleLoss_OT_FOA(EnsembleLoss):
    def __init__(self, extractors: list[ClipExtractor], cluster_number=5):
        super().__init__(extractors)
        self.global_features_gt = []
        self.local_features_gt = []
        self.prev_loss = []
        self.cluster_number = cluster_number

    @torch.no_grad()
    def set_ground_truth(self, x: Tensor):
        self.global_features_gt.clear()
        self.local_features_gt.clear()

        for model in self.extractors:
            global_f, local_f = model.global_local_features(x)
            self.global_features_gt.append(global_f)

            local_f = local_f.squeeze(0)
            local_f_center = get_cluster_center(local_f, self.cluster_number).unsqueeze(0)
            self.local_features_gt.append(local_f_center)

    def forward(self, global_features: list[Tensor], local_features: list[Tensor]):
        global_loss = []
        local_loss = []

        for index in range(len(self.extractors)):
            global_gt = self.global_features_gt[index]
            global_f = global_features[index].unsqueeze(0)
            g_loss = OT(global_gt, global_f)
            global_loss.append(g_loss)

            local_gt = self.local_features_gt[index].squeeze(0)
            local_f = local_features[index].squeeze(0)
            l_loss = OT(local_gt, local_f)
            local_loss.append(l_loss)

        total_losses = [
            global_loss[i] + 0.2 * local_loss[i]
            for i in range(len(self.extractors))
        ]
        return ComplexLoss(total_losses, self.prev_loss)