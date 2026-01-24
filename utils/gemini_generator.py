from PIL import Image
from google import genai

from .utils import load_api_keys

class GeminiGenerator:
    def __init__(self, model_name):
        self.model_name = model_name
        api = load_api_keys()
        self.client = genai.Client(api_key=api.gemini)
    
    def generate(self, prompt, image_path, max_tokens, thinking=False):
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[prompt, Image.open(image_path)],
        )
        if response.text:
            return response.text.strip()
        else:
            return ""