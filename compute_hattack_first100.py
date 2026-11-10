"""Read-only summary of first 100 rows from the existing DHRA 1000-image evaluation."""
import json
from pathlib import Path

for model in ('qwen3b', 'qwen7b', 'llava15', 'llava16'):
    path = Path('cache/evaluation/DHRA_NIPS1000') / model / 'judge_local_qwen7b/records.json'
    records = json.loads(path.read_text())
    values = [records[f'{i:04d}']['similarity'] for i in range(100)]
    print(model, len(values), sum(x >= 0.5 for x in values),
          round(sum(x >= 0.5 for x in values) / len(values), 6),
          round(sum(values) / len(values), 6))
