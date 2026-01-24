import torch
import random
from torch import Tensor

from .clip_extractor import ClipExtractor
from .function import OT, ComplexLoss, get_cluster_center
from .ensemble_extractor import EnsembleExtractor, EnsembleLoss

class EnsembleExtractor_random(EnsembleExtractor):
    def __init__(self, extractors: list[ClipExtractor],
        random_layer: list[int], random_index: list[float],
        index_in_layer: bool=True, cluster_number=5
    ):
        super().__init__(extractors)
        self.random_layer = random_layer
        self.random_index = random_index
        self.index_in_layer = index_in_layer
        self.cluster_number = cluster_number

        assert len(self.random_layer) == len(self.extractors), "Random layer list length must match extractors length."
        assert len(self.random_index) == len(self.extractors), "Random index list length must match extractors length."

    def forward(self, x: Tensor):
        global_features = []
        local_features = []
        global_features_random = []
        local_features_random = []

        for i, model in enumerate(self.extractors):

            index = self.random_index[i]
            if self.index_in_layer:
                index = (self.random_layer[i] + self.random_index[i]) / 3

            global_f, local_f, global_i, local_i = model.global_local_index_features(x, index)

            global_features.append(global_f.squeeze())
            global_features_random.append(global_i.squeeze())

            local_f_center = get_cluster_center(local_f[0], self.cluster_number).unsqueeze(0)
            local_features.append(local_f_center)
            local_i_center = get_cluster_center(local_i[0], self.cluster_number).unsqueeze(0)
            local_features_random.append(local_i_center)

        return global_features, local_features, global_features_random, local_features_random

class EnsembleLoss_random(EnsembleLoss):
    def __init__(self, extractors: list[ClipExtractor],
        random_layer: list[int], random_index: list[float],
        pin_layer: bool=False, pin_index: bool=False, index_in_layer: bool=True,
        cluster_number=5
    ):
        super().__init__(extractors)
        self.global_features_gt = []
        self.local_features_gt = []
        self.global_features_random_gt = []
        self.local_features_random_gt = []
        self.prev_loss = []

        self.random_layer = random_layer
        self.random_index = random_index
        self.pin_layer = pin_layer
        self.pin_index = pin_index
        self.index_in_layer = index_in_layer
        self.cluster_number = cluster_number

        assert len(self.random_layer) == len(self.extractors), "Random layer list length must match extractors length."
        assert len(self.random_index) == len(self.extractors), "Random index list length must match extractors length."

    @torch.no_grad()
    def set_ground_truth(self, x: Tensor):
        self.global_features_gt.clear()
        self.local_features_gt.clear()
        self.global_features_random_gt.clear()
        self.local_features_random_gt.clear()

        if not self.pin_layer:
            random.shuffle(self.random_layer)

        for i, model in enumerate(self.extractors):

            if not self.pin_index:
                self.random_index[i] = random.random()
            
            index = self.random_index[i]
            if self.index_in_layer:
                index = (self.random_layer[i] + self.random_index[i]) / 3

            global_f, local_f, global_i, local_i = model.global_local_index_features(x, index)
            self.global_features_gt.append(global_f)
            self.global_features_random_gt.append(global_i)

            local_f = local_f.squeeze(0)
            local_f_center = get_cluster_center(local_f, self.cluster_number).unsqueeze(0)
            self.local_features_gt.append(local_f_center)

            self.local_features_random_gt.append(local_i)   # 未进行聚类

    def forward(self,
        global_features: list[Tensor],
        local_features: list[Tensor],
        global_features_random: list[Tensor],
        local_features_random: list[Tensor]
    ) -> Tensor:
        global_loss = []
        local_loss = []
        global_random_loss = []
        local_random_loss = []

        for index in range(len(self.extractors)):
            global_gt = self.global_features_gt[index]
            global_f = global_features[index].unsqueeze(0)
            g_loss = OT(global_gt, global_f)
            global_loss.append(g_loss)

            local_gt = self.local_features_gt[index].squeeze(0)
            local_f = local_features[index].squeeze(0)
            l_loss = OT(local_gt, local_f)
            local_loss.append(l_loss)

            global_i_gt = self.global_features_random_gt[index]
            global_i = global_features_random[index].unsqueeze(0)
            g_i_loss = OT(global_i_gt, global_i)
            global_random_loss.append(g_i_loss)

            local_i_gt = self.local_features_random_gt[index].squeeze(0)
            local_i = local_features_random[index].squeeze(0)
            l_i_loss = OT(local_i_gt, local_i)
            local_random_loss.append(l_i_loss)

        total_losses = [
            global_loss[i]
            + 0.2 * local_loss[i]
            + 0.2 * global_random_loss[i]
            + 0.2 * local_random_loss[i]
            for i in range(len(self.extractors))
        ]
        return ComplexLoss(total_losses, self.prev_loss)

class EnsembleExtractor_random_non_local(EnsembleExtractor):
    def __init__(self, extractors: list[ClipExtractor],
        random_layer: list[int], random_index: list[float], index_in_layer: bool=True
    ):
        super().__init__(extractors)
        self.random_layer = random_layer
        self.random_index = random_index
        self.index_in_layer = index_in_layer

        assert len(self.random_layer) == len(self.extractors), "Random layer list length must match extractors length."
        assert len(self.random_index) == len(self.extractors), "Random index list length must match extractors length."

    def forward(self, x: Tensor):
        global_features = []
        global_features_random = []

        for i, model in enumerate(self.extractors):

            index = self.random_index[i]
            if self.index_in_layer:
                index = (self.random_layer[i] + self.random_index[i]) / 3

            global_f, global_i = model.global_index_features(x, index)
            global_features.append(global_f.squeeze())
            global_features_random.append(global_i.squeeze())

        return global_features, global_features_random

class EnsembleLoss_random_non_local(EnsembleLoss):
    def __init__(self, extractors: list[ClipExtractor],
        random_layer: list[int], random_index: list[float],
        pin_layer: bool=False, pin_index: bool=False, index_in_layer: bool=True
    ):
        super().__init__(extractors)
        self.global_features_gt = []
        self.local_features_gt = []
        self.global_features_random_gt = []
        self.local_features_random_gt = []
        self.prev_loss = []

        self.random_layer = random_layer
        self.random_index = random_index
        self.pin_layer = pin_layer
        self.pin_index = pin_index
        self.index_in_layer = index_in_layer

        assert len(self.random_layer) == len(self.extractors), "Random layer list length must match extractors length."
        assert len(self.random_index) == len(self.extractors), "Random index list length must match extractors length."

    @torch.no_grad()
    def set_ground_truth(self, x: Tensor):
        self.global_features_gt.clear()
        self.local_features_gt.clear()
        self.global_features_random_gt.clear()
        self.local_features_random_gt.clear()
        
        if not self.pin_layer:
            random.shuffle(self.random_layer)

        for i, model in enumerate(self.extractors):

            if not self.pin_index:
                self.random_index[i] = random.random()
            
            index = self.random_index[i]
            if self.index_in_layer:
                index = (self.random_layer[i] + self.random_index[i]) / 3

            global_f, global_i = model.global_index_features(x, index)
            self.global_features_gt.append(global_f)
            self.global_features_random_gt.append(global_i)

    def forward(self, global_features: list[Tensor], global_features_random: list[Tensor]):
        global_loss = []
        global_loss_random= []

        for index in range(len(self.extractors)):
            global_gt = self.global_features_gt[index]
            global_f = global_features[index].unsqueeze(0)
            g_loss = OT(global_gt, global_f)
            global_loss.append(g_loss)

            global_i_gt = self.global_features_random_gt[index]
            global_i = global_features_random[index].unsqueeze(0)
            g_i_loss = OT(global_i_gt, global_i)
            global_loss_random.append(g_i_loss)

        total_losses = [
            global_loss[i]
            + 0.2 * global_loss_random[i]
            for i in range(len(self.extractors))
        ]
        return ComplexLoss(total_losses, self.prev_loss)