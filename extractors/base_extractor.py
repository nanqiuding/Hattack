import torch.nn as nn
from torch import Tensor
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

class ClipExtractor(nn.Module):
    model: CLIPModel
    processor: CLIPProcessor
    normalizer: transforms.Compose

    def text_features(self, texts: list[str]):
        text_inputs = self.processor(
            text=texts, return_tensors="pt",
            padding=True, truncation=True
        ).to(self.model.device)
        text_outputs = self.model.text_model(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"]
        )
        text_embeds = text_outputs.last_hidden_state[:, 0, :]
        return text_embeds

    def global_features(self, x: Tensor) -> tuple[Tensor, ...]:
        inputs = dict(pixel_values=self.normalizer(x))
        inputs["pixel_values"] = inputs["pixel_values"]
        outputs = self.model.vision_model(pixel_values=inputs["pixel_values"])
        features = outputs.last_hidden_state

        global_feature = features[:, 0, :]
        global_feature = global_feature / global_feature.norm(dim=1, keepdim=True)

        return global_feature,

    def global_local_features(self, x: Tensor) -> tuple[Tensor, ...]:
        inputs = dict(pixel_values=self.normalizer(x))
        inputs["pixel_values"] = inputs["pixel_values"]
        outputs = self.model.vision_model(pixel_values=inputs["pixel_values"])
        features = outputs.last_hidden_state

        global_feature = features[:, 0, :]
        global_feature = global_feature / global_feature.norm(dim=1, keepdim=True)
        local_feature = features[:, 1:, :]
        local_feature = local_feature / local_feature.norm(dim=1, keepdim=True)

        return global_feature, local_feature

    def global_local_middle_features(self, x: Tensor) -> tuple[Tensor, ...]:
        inputs = dict(pixel_values=self.normalizer(x))
        inputs["pixel_values"] = inputs["pixel_values"]
        outputs = self.model.vision_model(pixel_values=inputs["pixel_values"], output_hidden_states=True, output_attentions=True)
        features_middle = outputs.hidden_states[int(len(outputs.hidden_states)/2)]
        features = outputs.last_hidden_state

        global_feature = features[:, 0, :]
        global_feature = global_feature / global_feature.norm(dim=1, keepdim=True)
        local_feature = features[:, 1:, :]
        local_feature = local_feature / local_feature.norm(dim=1, keepdim=True)

        global_feature_middle = features_middle[:, 0, :]
        global_feature_middle = global_feature_middle / global_feature_middle.norm(dim=1, keepdim=True)
        local_features_middle = features_middle[:, 1:, :]
        local_features_middle = local_features_middle / local_features_middle.norm(dim=1, keepdim=True)
        return global_feature, local_feature, global_feature_middle, local_features_middle
    
    def global_middle_features(self, x: Tensor) -> tuple[Tensor, ...]:
        inputs = dict(pixel_values=self.normalizer(x))
        inputs["pixel_values"] = inputs["pixel_values"]
        outputs = self.model.vision_model(pixel_values=inputs["pixel_values"], output_hidden_states=True)
        features_middle = outputs.hidden_states[int(len(outputs.hidden_states)/2)]
        features = outputs.last_hidden_state

        global_feature = features[:, 0, :]
        global_feature = global_feature / global_feature.norm(dim=1, keepdim=True)

        global_feature_middle = features_middle[:, 0, :]
        global_feature_middle = global_feature_middle / global_feature_middle.norm(dim=1, keepdim=True)

        return global_feature, global_feature_middle

    def global_index_features(self, x: Tensor, i: float) -> tuple[Tensor, ...]:
        inputs = dict(pixel_values=self.normalizer(x))
        inputs["pixel_values"] = inputs["pixel_values"]
        outputs = self.model.vision_model(pixel_values=inputs["pixel_values"], output_hidden_states=True)

        features = outputs.last_hidden_state
        global_feature = features[:, 0, :]
        global_feature = global_feature / global_feature.norm(dim=1, keepdim=True)

        i = int(i * len(outputs.hidden_states))
        features_i = outputs.hidden_states[i]
        global_feature_i = features_i[:, 0, :]
        global_feature_i = global_feature_i / global_feature_i.norm(dim=1, keepdim=True)

        return global_feature, global_feature_i

    def global_local_index_features(self, x: Tensor, i: float) -> tuple[Tensor, ...]:
        inputs = dict(pixel_values=self.normalizer(x))
        inputs["pixel_values"] = inputs["pixel_values"]
        outputs = self.model.vision_model(pixel_values=inputs["pixel_values"], output_hidden_states=True)

        features_last = outputs.last_hidden_state
        global_feature_last = features_last[:, 0, :]
        global_feature_last = global_feature_last / global_feature_last.norm(dim=1, keepdim=True)
        local_feature_last = features_last[:, 1:, :]
        local_feature_last = local_feature_last / local_feature_last.norm(dim=1, keepdim=True)

        i = int(i * len(outputs.hidden_states))
        features_i = outputs.hidden_states[i]
        global_feature_i = features_i[:, 0, :]
        global_feature_i = global_feature_i / global_feature_i.norm(dim=1, keepdim=True)
        local_feature_i = features_i[:, 1:, :]
        local_feature_i = local_feature_i / local_feature_i.norm(dim=1, keepdim=True)

        return global_feature_last, local_feature_last, global_feature_i, local_feature_i