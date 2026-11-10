"""Summarize a fixed prefix of already judged records without changing originals."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--method', required=True)
p.add_argument('--model', required=True)
p.add_argument('--limit', type=int, default=100)
p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
records_path = Path('cache/evaluation') / args.method / args.model / 'judge_local_qwen7b/records.json'
records = json.loads(records_path.read_text())
values = [float(records[f'{i:04d}']['similarity']) for i in range(args.limit)]
summary = {'method': args.method, 'caption_model': args.model,
           'judge': 'local Qwen2.5-VL-7B-Instruct (text-only)',
           'not_comparable_to_paper_gpt4o_judge': True,
           'samples': len(values), 'threshold': 0.5,
           'successes': sum(x >= 0.5 for x in values),
           'ASR': sum(x >= 0.5 for x in values) / len(values),
           'AvgSim': sum(values) / len(values),
           'derived_from': str(records_path)}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(summary, indent=2))
print('SUBSET_SUMMARY', args.output, summary['ASR'], summary['AvgSim'])
