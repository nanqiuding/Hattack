import re
import json
from openai import OpenAI

from .utils import load_api_keys

matcher_prompt = """You will be performing a keyword-matching task. You will be given a short description and a list of keywords. Your goal is to find matches between the keywords and the content in the description.

Here is the description text:
<description>
{description}
</description>

Here is the list of keywords:
<keywords>
{keywords}
</keywords>

For each keyword in the list, follow these steps:
1. Look for an exact match of the keyword in the description text.
2. If an exact match is not found, look for words or phrases with similar meanings to the keyword. For example, 'bite' could match with 'chew', or 'snow-covered' could match with 'snow'.
3. If you find a match (either exact or similar), record the keyword and its matched content.

Your output should be in JSON format, where each key is a keyword from the list, and its value is the matched content from the description. Only include keywords that have matches. For example:

{{
  "bite": "chew",
  "snow": "snow-covered"
}}

Here are some important points to remember:
- Only include keywords that have matches in the description.
- If a keyword doesn't have a match, do not include it in the JSON.
- The matched content should be the exact text from the description, not a paraphrase.
- If there are multiple matches for a keyword, use the most relevant or closest match.

Please provide your answer in the following format:
<answer>
{{
  // Your JSON output here
}}
</answer>

Remember to only include the JSON in your answer, with no additional explanation or text."""

def extract_answer_regex(s):
    pattern = r"<\s*answer\s*>([\s\S]*?)<\s*\/\s*answer\s*>"
    match = re.search(pattern, s, re.IGNORECASE | re.DOTALL)
    
    if match:
        content = match.group(1).strip()
        return json.loads(content)
    else:
        return None

class KeywordMatcher:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        self.client = OpenAI(api_key=load_api_keys().gpt)
    
    def _gen(self, prompt: str):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()

    def match(self, desc: str, keywords: list[str]) -> dict:
        match_result = self._gen(matcher_prompt.format(
            description = desc,
            keywords = json.dumps(keywords)
        ))
        return extract_answer_regex(match_result)
