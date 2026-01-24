import torch
from torch import nn, Tensor

from .clip_extractor import ClipExtractor
from .function import OT, ComplexLoss, get_cluster_center

class EnsembleExtractor(nn.Module):
    def __init__(self, extractors: list[ClipExtractor]):
        super().__init__()
        self.extractors = extractors

    def forward(self, x):
        features = {}
        for i, model in enumerate(self.extractors):
            features[i] = model(x)
        return features

class EnsembleLoss(nn.Module):
    def __init__(self, extractors: list[ClipExtractor]):
        super().__init__()
        self.extractors = extractors

    @torch.no_grad()
    def set_ground_truth(self, x: Tensor):
        self.ground_truth = []
        for model in self.extractors:
            self.ground_truth.append(model(x))

    def forward(self, features: list[Tensor]):
        loss = 0
        for index in range(len(self.extractors)):
            feature = features[index]
            gt = self.ground_truth[index]
            loss += torch.mean(torch.sum(feature * gt, dim=1))
            
        loss = loss / len(self.extractors)
        return loss

# ==================== unused ====================

class EnsembleExtractor_OT3(EnsembleExtractor):
    def __init__(self, extractors: list[ClipExtractor]):
        super().__init__(extractors)

    def forward(self, x: Tensor):
        global_features = []
        local_features = []
        
        for model in self.extractors:
            global_f, local_f = model.global_local_features(x)
            global_features.append(global_f.squeeze())

            local_f_center = get_cluster_center(local_f[0], 3).unsqueeze(0)
            local_features.append(local_f_center)

        return global_features, local_features

class EnsembleLoss_OT(EnsembleLoss):
    def __init__(self, extractors: list[ClipExtractor]):
        super().__init__(extractors)
        self.global_features_gt = []
        self.local_features_gt = []

    @torch.no_grad()
    def set_ground_truth(self, x: Tensor):
        self.global_features_gt.clear()
        self.local_features_gt.clear()

        for model in self.extractors:
            global_f, local_f = model.global_local_features(x)
            self.global_features_gt.append(global_f)

            local_f = local_f.squeeze(0)
            local_f_center = get_cluster_center(local_f).unsqueeze(0)
            self.local_features_gt.append(local_f_center)

    def forward(self, global_features: list[Tensor], local_features: list[Tensor]):
        global_loss = 0
        local_loss = 0

        for index in range(len(self.extractors)):
            global_gt = self.global_features_gt[index]
            global_f = global_features[index]
            global_loss += torch.mean(torch.sum(global_f * global_gt, dim=1))

            local_gt = self.local_features_gt[index].squeeze(0)
            local_f = local_features[index].squeeze(0)
            local_loss += OT(local_gt, local_f) * 2

        global_loss = global_loss / len(self.extractors)
        local_loss = local_loss / len(self.extractors)

        return global_loss + local_loss * 0.1

class EnsembleLoss_OT_Auto(EnsembleLoss):
    def __init__(self, extractors: list[ClipExtractor]):
        super().__init__(extractors)
        self.global_features_gt = []
        self.local_features_gt = []
        self.prev_loss = []

    @torch.no_grad()
    def set_ground_truth(self, x: Tensor):
        self.global_features_gt.clear()
        self.local_features_gt.clear()

        for model in self.extractors:
            global_f, local_f = model.global_local_features(x)
            self.global_features_gt.append(global_f)

            local_f = local_f.squeeze(0)
            local_f_center = get_cluster_center(local_f).unsqueeze(0)
            self.local_features_gt.append(local_f_center)

    def forward(self, global_features: list[Tensor], local_features: list[Tensor]):
        global_loss = []
        local_loss = []

        for index in range(len(self.extractors)):
            global_gt = self.global_features_gt[index]
            global_f = global_features[index]
            g_loss = torch.mean(torch.sum(global_f * global_gt, dim=1))

            local_gt = self.local_features_gt[index].squeeze(0)
            local_f = local_features[index].squeeze(0)
            l_loss = OT(local_gt, local_f) * 2

            global_loss.append(g_loss)
            local_loss.append(l_loss)

        total_losses = [
            global_loss[i] + 0.1 * local_loss[i]
            for i in range(len(self.extractors))
        ]
        return ComplexLoss(total_losses, self.prev_loss)

class EnsembleLoss_OT_ablation_wo_local(EnsembleLoss):
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

    def forward(self, global_features: list[Tensor]):
        global_loss = []

        for index in range(len(self.extractors)):
            global_gt = self.global_features_gt[index]
            global_f = global_features[index].unsqueeze(0)
            g_loss = OT(global_gt, global_f)
            global_loss.append(g_loss)

        total_losses = [
            global_loss[i]
            for i in range(len(self.extractors))
        ]
        return ComplexLoss(total_losses, self.prev_loss)

class EnsembleLoss_OT_ablation_wo_dynamic(EnsembleLoss):
    def __init__(self, extractors: list[ClipExtractor], cluster_number=5):
        super().__init__(extractors)
        self.global_features_gt = []
        self.local_features_gt = []
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

    def forward(self, global_featues: list[Tensor]):
        global_loss = []

        for index in range(len(self.extractors)):
            global_gt = self.global_features_gt[index]
            global_f = global_featues[index].unsqueeze(0)
            g_loss = OT(global_gt, global_f)
            global_loss.append(g_loss)

        total_losses = [
            global_loss[i]
            for i in range(len(self.extractors))
        ]

        total_loss = sum(total_losses) / len(total_losses)
        return total_loss