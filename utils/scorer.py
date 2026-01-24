from openai import OpenAI

from .utils import load_api_keys

scorer_prompt = """Rate the semantic similarity between the following two texts on a scale from 0 to 1.
        
**Criteria for similarity measurement:**
1. **Main Subject Consistency:** If both descriptions refer to the same key subject or object (e.g., a person, food, an event), they should receive a higher similarity score.
2. **Relevant Description**: If the descriptions are related to the same context or topic, they should also contribute to a higher similarity score.
3. **Ignore Fine-Grained Details:** Do not penalize differences in **phrasing, sentence structure, or minor variations in detail**. Focus on **whether both descriptions fundamentally describe the same thing.**
4. **Partial Matches:** If one description contains extra information but does not contradict the other, they should still have a high similarity score.
5. **Similarity Score Range:** 
    - **1.0**: Nearly identical in meaning.
    - **0.8-0.9**: Same subject, with highly related descriptions.
    - **0.7-0.8**: Same subject, core meaning aligned, even if some details differ.
    - **0.5-0.7**: Same subject but different perspectives or missing details.
    - **0.3-0.5**: Related but not highly similar (same general theme but different descriptions).
    - **0.0-0.2**: Completely different subjects or unrelated meanings.
    
Text 1: {}
Text 2: {}

Output only a single number between 0 and 1. Do not include any explanation or additional text."""

class GPTScorer:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        self.client = OpenAI(api_key=load_api_keys().gpt)

    def compute_similarity(self, text1: str, text2: str) -> float:
        prompt = scorer_prompt.format(text1, text2)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            top_p=1,
            max_tokens=100,
        )
        score = response.choices[0].message.content.strip()
        return min(1.0, max(0.0, float(score)))