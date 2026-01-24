import torch
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

from .base_extractor import ClipExtractor

model_cache_dir = "cache/models/"

ClipB16_model_name = "openai/clip-vit-base-patch16"
ClipB32_model_name = "openai/clip-vit-base-patch32"
Laion_model_name = "laion/CLIP-ViT-G-14-laion2B-s12B-b42K"

class ClipB16Extractor(ClipExtractor):
    def __init__(self):
        super().__init__()
        self.model = CLIPModel.from_pretrained(
            ClipB16_model_name,
            cache_dir=model_cache_dir + ClipB16_model_name
        )
        self.processor = CLIPProcessor.from_pretrained(
            ClipB16_model_name,
            cache_dir=model_cache_dir + ClipB16_model_name
        )
        self.normalizer = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
            transforms.CenterCrop(224),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

class ClipB32Extractor(ClipExtractor):
    def __init__(self):
        super().__init__()
        self.model = CLIPModel.from_pretrained(
            ClipB32_model_name,
            cache_dir=model_cache_dir + ClipB32_model_name
        )
        self.processor = CLIPProcessor.from_pretrained(
            ClipB32_model_name,
            cache_dir=model_cache_dir + ClipB32_model_name
        )
        self.normalizer = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
            transforms.CenterCrop(224),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

class LaionExtractor(ClipExtractor):
    def __init__(self):
        super().__init__()
        self.model = CLIPModel.from_pretrained(
            Laion_model_name,
            cache_dir=model_cache_dir + Laion_model_name
        )
        self.processor = CLIPProcessor.from_pretrained(
            Laion_model_name,
            cache_dir=model_cache_dir + Laion_model_name
        )
        self.normalizer = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
            transforms.CenterCrop(224),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])