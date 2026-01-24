import torch
from torch import Tensor

from .clip_extractor import ClipExtractor
from .function import OT, ComplexLoss, get_cluster_center
from .ensemble_extractor import EnsembleExtractor, EnsembleLoss

class EnsembleExtractor_middle(EnsembleExtractor):
    def __init__(self, extractors: list[ClipExtractor], cluster_number=5):
        super().__init__(extractors)
        self.cluster_number = cluster_number

    def forward(self, x: Tensor):
        global_features = []
        local_features = []
        global_features_middle = []
        local_features_middle = []

        for model in self.extractors:
            global_f, local_f, global_middle, local_middle = model.global_local_middle_features(x)
            global_features.append(global_f.squeeze())
            global_features_middle.append(global_middle.squeeze())

            local_f_center = get_cluster_center(local_f[0], self.cluster_number).unsqueeze(0)
            local_features.append(local_f_center)
            local_middle_center = get_cluster_center(local_middle[0], self.cluster_number).unsqueeze(0)
            local_features_middle.append(local_middle_center)

        return global_features, local_features, global_features_middle, local_features_middle

class EnsembleLoss_middle(EnsembleLoss):
    def __init__(self, extractors: list[ClipExtractor], cluster_number=5):
        super().__init__(extractors)
        self.global_features_gt = []
        self.local_features_gt = []
        self.global_features_middle_gt = []
        self.local_features_middle_gt = []
        self.prev_loss = []
        self.cluster_number = cluster_number

    @torch.no_grad()
    def set_ground_truth(self, x: Tensor):
        self.global_features_gt.clear()
        self.local_features_gt.clear()
        self.global_features_middle_gt.clear()
        self.local_features_middle_gt.clear()

        for model in self.extractors:
            global_f, local_f, global_middle, local_middle = model.global_local_middle_features(x)
            self.global_features_gt.append(global_f)
            self.global_features_middle_gt.append(global_middle)

            local_f = local_f.squeeze(0)
            local_f_center = get_cluster_center(local_f, self.cluster_number).unsqueeze(0)
            self.local_features_gt.append(local_f_center)

            self.local_features_middle_gt.append(local_middle)    # 未进行聚类？

    def forward(
        self, global_features: list[Tensor], local_features: list[Tensor],
        global_features_middle: list[Tensor], local_features_middle: list[Tensor]
    ):
        global_loss = []
        local_loss = []
        global_middle_loss = []
        local_middle_loss = []

        for index in range(len(self.extractors)):
            global_gt = self.global_features_gt[index]
            global_f = global_features[index].unsqueeze(0)
            g_loss = OT(global_gt, global_f)
            global_loss.append(g_loss)

            local_gt = self.local_features_gt[index].squeeze(0)
            local_f = local_features[index].squeeze(0)
            l_loss = OT(local_gt, local_f)
            local_loss.append(l_loss)

            global_middle_gt = self.global_features_middle_gt[index]
            global_middle = global_features_middle[index].unsqueeze(0)
            g_middle_loss = OT(global_middle_gt, global_middle)
            global_middle_loss.append(g_middle_loss)

            local_middle_gt = self.local_features_middle_gt[index].squeeze(0)
            local_middle = local_features_middle[index].squeeze(0)
            l_middle_loss = OT(local_middle_gt, local_middle)
            local_middle_loss.append(l_middle_loss)

        total_losses = [
            global_loss[i]
            + 0.2 * local_loss[i]
            + 0.2 * global_middle_loss[i]
            + 0.2 * local_middle_loss[i]
            for i in range(len(self.extractors))
        ]
        return ComplexLoss(total_losses, self.prev_loss)

class EnsembleExtractor_middle_non_local(EnsembleExtractor):
    def __init__(self, extractors: list[ClipExtractor], cluster_number=5):
        super().__init__(extractors)
        self.cluster_number = cluster_number

    def forward(self, x: Tensor):
        global_features = []
        global_features_middle = []

        for model in self.extractors:
            global_f, global_middle = model.global_middle_features(x)
            global_features.append(global_f.squeeze())
            global_features_middle.append(global_middle.squeeze())

        return global_features, global_features_middle

class EnsembleLoss_middle_non_local(EnsembleLoss):
    def __init__(self, extractors: list[ClipExtractor], cluster_number=5):
        super().__init__(extractors)
        self.global_features_gt = []
        self.global_features_middle_gt = []
        self.prev_loss = []
        self.cluster_number = cluster_number

    @torch.no_grad()
    def set_ground_truth(self, x: Tensor):
        self.global_features_gt.clear()
        self.global_features_middle_gt.clear()

        for model in self.extractors:
            global_f, global_middle= model.global_middle_features(x)
            self.global_features_gt.append(global_f)
            self.global_features_middle_gt.append(global_middle)

    def forward(self, global_features: list[Tensor], global_middle_features: list[Tensor]):
        global_loss = []
        global_middle_loss = []

        for index in range(len(self.extractors)):
            global_gt = self.global_features_gt[index]
            global_f = global_features[index].unsqueeze(0)
            g_loss = OT(global_gt, global_f)
            global_loss.append(g_loss)

            global_middle_gt = self.global_features_middle_gt[index]
            global_middle = global_middle_features[index].unsqueeze(0)
            g_middle_loss = OT(global_middle_gt, global_middle)
            global_middle_loss.append(g_middle_loss)

        total_losses = [
            global_loss[i]
            + 0.2 * global_middle_loss[i]
            for i in range(len(self.extractors))
        ]
        return ComplexLoss(total_losses, self.prev_loss)
