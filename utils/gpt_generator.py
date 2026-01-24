from openai import OpenAI

from .utils import load_api_keys, encode_image

class GPTGenerator:
    def __init__(self, model_name):
        self.model_name = model_name
        api = load_api_keys()
        self.client = OpenAI(api_key=api.gpt, timeout=30)
    
    def generate(self, prompt, image_path, max_tokens, thinking=False):
        base64_image = encode_image(image_path)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{
                "role": "user",
                "content": [
                    { "type": "text", "text": prompt },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ],
            }],
        )
        return response.choices[0].message.content