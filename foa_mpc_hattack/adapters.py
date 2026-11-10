"""Differentiable image-feature adapters for the FOA/MPC experiment.

FOA CLIP classes are copied unchanged under vendor/foa_official.  MPC's
checkpoint IDs and image normalization are used for DINOv2 and InternVL3.
"""
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .vendor.foa_official.surrogates.FeatureExtractors.ClipB16 import ClipB16FeatureExtractor
from .vendor.foa_official.surrogates.FeatureExtractors.ClipB32 import ClipB32FeatureExtractor
from .vendor.foa_official.surrogates.FeatureExtractors.ClipLaion import ClipLaionFeatureExtractor

CLIP_CLASSES = {
    "CLIP_B16": ClipB16FeatureExtractor,
    "CLIP_B32": ClipB32FeatureExtractor,
    "CLIP_LAION": ClipLaionFeatureExtractor,
}
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass
class SurrogateFeatures:
    global_feat: torch.Tensor
    patch_feats: Optional[torch.Tensor] = None
    layer_cls: Optional[torch.Tensor] = None
    valid_patch_mask: Optional[torch.Tensor] = None


class SurrogateAdapter(nn.Module):
    def forward_features(self, images_01: torch.Tensor, need_patches=False,
                         need_layers=False) -> SurrogateFeatures:
        raise NotImplementedError


class FOAClipAdapter(SurrogateAdapter):
    """Expose official FOA hidden CLS/patch tokens; projected CLIP is distinct."""
    def __init__(self, name, device):
        super().__init__()
        self.name = name
        self.extractor = CLIP_CLASSES[name]().eval().to(device).requires_grad_(False)

    def forward_features(self, images_01, need_patches=False, need_layers=False):
        if images_01.ndim != 4 or images_01.shape[1] != 3:
            raise ValueError("Expected [B,3,H,W]")
        pixels = self.extractor.normalizer(images_01.clamp(0, 1) * 255)
        outputs = self.extractor.model.vision_model(
            pixel_values=pixels, output_hidden_states=need_layers)
        hidden = outputs.last_hidden_state
        global_feat = F.normalize(hidden[:, 0].float(), dim=-1)
        patches = F.normalize(hidden[:, 1:].float(), dim=-1) if need_patches else None
        layers = (torch.stack([state[:, 0].float() for state in outputs.hidden_states[1:]], 1)
                  if need_layers else None)
        return SurrogateFeatures(global_feat, patches, layers)


class DINOv2Adapter(SurrogateAdapter):
    checkpoint_id = "facebook/dinov2-base"
    def __init__(self, device):
        super().__init__()
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(self.checkpoint_id).eval().to(device)
        self.model.requires_grad_(False)

    def forward_features(self, images_01, need_patches=False, need_layers=False):
        pixels = F.interpolate(images_01.clamp(0, 1), size=(224, 224), mode="bicubic", align_corners=False)
        mean = pixels.new_tensor(CLIP_MEAN)[None, :, None, None]
        std = pixels.new_tensor(CLIP_STD)[None, :, None, None]
        outputs = self.model(pixel_values=(pixels - mean) / std,
                             output_hidden_states=need_layers)
        hidden = outputs.last_hidden_state
        layers = (torch.stack([state[:, 0].float() for state in outputs.hidden_states[1:]], 1)
                  if need_layers else None)
        return SurrogateFeatures(F.normalize(hidden[:, 0].float(), dim=-1),
                                 F.normalize(hidden[:, 1:].float(), dim=-1) if need_patches else None,
                                 layers)


class InternVL3Adapter(SurrogateAdapter):
    checkpoint_id = "OpenGVLab/InternVL3-1B-hf"
    def __init__(self, device):
        super().__init__()
        from transformers import AutoModelForImageTextToText
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.checkpoint_id, torch_dtype=torch.bfloat16).eval().to(device)
        self.model.requires_grad_(False)

    def forward_features(self, images_01, need_patches=False, need_layers=False):
        # For the square 224x224 attack input, MPC's dynamic tiler selects one
        # 448x448 tile.  Interpolation remains differentiable to attack pixels.
        pixels = F.interpolate(images_01.clamp(0, 1), size=(448, 448), mode="bilinear", align_corners=False)
        mean = pixels.new_tensor(CLIP_MEAN)[None, :, None, None]
        std = pixels.new_tensor(CLIP_STD)[None, :, None, None]
        pixels = ((pixels - mean) / std).to(dtype=torch.bfloat16)
        features = self.model.get_image_features(pixel_values=pixels)
        if isinstance(features, (tuple, list)):
            features = features[0]
        if features.ndim == 3:
            global_feat = features.float().mean(1)
        elif features.ndim == 2:
            global_feat = features.float()
        else:
            raise ValueError(f"Unexpected InternVL image features: {features.shape}")
        # InternVisionModel uses mean pooling rather than a CLS representation
        # for image tokens. Use the mean of patch tokens at each visual layer.
        layers = None
        if need_layers:
            pooled = []
            def capture(_module, _inputs, output):
                state = output[0] if isinstance(output, (tuple, list)) else output
                pooled.append(state[:, 1:].float().mean(1))
            handles = [layer.register_forward_hook(capture)
                       for layer in self.model.vision_tower.encoder.layer]
            try:
                self.model.vision_tower(pixel_values=pixels)
            finally:
                for handle in handles:
                    handle.remove()
            layers = torch.stack(pooled, 1)
        return SurrogateFeatures(F.normalize(global_feat, dim=-1), None, layers)
