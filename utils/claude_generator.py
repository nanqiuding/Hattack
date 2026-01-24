import anthropic

from .utils import load_api_keys, encode_image, get_media_type

class ClaudeGenerator:
    def __init__(self, model_name):
        self.model_name = model_name
        api = load_api_keys()
        self.client = anthropic.Anthropic(api_key=api.claude, timeout=30)
    
    def generate(self, prompt, image_path, max_tokens, thinking=False):
        base64_image = encode_image(image_path)
        media_type = get_media_type(image_path)
        response = self.client.messages.create(
            model=self.model_name,
            messages=[{
                "role": "user",
                "content": [
                    { "type": "text", "text": prompt },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": base64_image,
                        },
                    },
                ],
            }],
            max_tokens=max_tokens + (1024 if thinking else 0),
            thinking={"type": "enabled", "budget_tokens": 1024} if thinking else None,
        )
        if thinking:
            return response.content[1].text
        else:
            return response.content[0].text