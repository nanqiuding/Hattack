import requests

from .gpt_generator import GPTGenerator
from .gemini_generator import GeminiGenerator
from .claude_generator import ClaudeGenerator

class ImageDescriptionGeneratorOpen:
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    def generate_description(self, image_path: str) -> str:
        files = { "image": open(image_path, "rb") }
        data = {
            "prompt": "Describe this image, no longer than 25 words.",
            "model": self.model_name
        }
        response = requests.post(
            url="http://localhost:8000/generate",
            files=files, data=data,
            timeout=30
        )
        return response.json()["description"]

generator_map = {
    "gpt-4o":   GPTGenerator,
    "gpt-4.1":  GPTGenerator,
    "o3":       GPTGenerator,
    
    "gemini-2.0-flash": GeminiGenerator,
    "gemini-2.5-pro":   GeminiGenerator,

    "claude-sonnet-4-5":    ClaudeGenerator,
}

class ImageDescriptionGeneratorClose:
    def __init__(self, model_name: str, thinking=False):
        self.genertor = generator_map[model_name](model_name)
        self.thinking = thinking

    def generate_description(self, image_path: str) -> str:
        return self.genertor.generate(
            "Describe this image in one concise sentence, no longer than 20 words.",
            image_path, 128, self.thinking
        )