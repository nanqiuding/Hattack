"""Collect completed P0 ablation evaluations without inventing missing scores."""
import argparse
import csv
import json
from pathlib import Path

GROUPS = {
    'core': [('A0', 'Base only'), ('A1', 'Base + Quantum Fidelity'),
             ('A2', 'Base + Decoherence'), ('A3', 'Full method')],
    'alignment': [('A1', 'Quantum Fidelity'), ('B1', 'Frobenius Distance'),
                  ('B2', 'Gram Cosine'), ('B3', 'Layer-wise Cosine')],
    'decoherence': [('A1', 'None'), ('C1', 'Mean only'),
                    ('C2', 'Variance only'), ('A3', 'Mean + Variance')],
}
MODELS = ('qwen3b', 'qwen7b', 'llava15', 'llava16')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=('stage20', 'stage100', 'full'), required=True)
    args = p.parse_args()
    root = Path('cache/ablation')
    output = root / args.stage
    output.mkdir(parents=True, exist_ok=True)
    variants = sorted({v for group in GROUPS.values() for v, _ in group})
    configs = {}
    for variant in variants:
        path = root / f'DHRA_ABL_{variant}' / 'config.json'
        if path.is_file():
            configs[variant] = json.loads(path.read_text())
            if args.stage in ('stage20', 'stage100'):
                configs[variant]['samples'] = 20 if args.stage == 'stage20' else 100
    (output / 'ablation_config.json').write_text(json.dumps(configs, indent=2))

    with (output / 'loss_scale_diagnostics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['variant', 'first_step_base', 'first_step_alignment', 'first_step_dec_mean',
                         'first_step_dec_variance', 'base_grad_l2', 'alignment_grad_l2',
                         'dec_mean_grad_l2', 'dec_variance_grad_l2', 'total_grad_l2'])
        for variant in variants:
            record = root / f'DHRA_ABL_{variant}' / '0000.json'
            if not record.is_file():
                continue
            first = json.loads(record.read_text())['history'][0]
            grads = first.get('component_grad_l2', {})
            writer.writerow([variant, first['base'], first['align'], first['dec_mean'],
                             first['dec_variance'], grads.get('base', 0), grads.get('align', 0),
                             grads.get('dec_mean', 0), grads.get('dec_variance', 0),
                             first['total_grad_l2']])

    summaries = {}
    for variant in variants:
        for model in MODELS:
            if args.stage in ('stage20', 'stage100'):
                path = output / variant / model / 'summary.json'
            else:
                path = Path('cache/evaluation') / f'DHRA_ABL_{variant}' / model / 'judge_local_qwen7b' / 'summary.json'
            if path.is_file():
                data = json.loads(path.read_text())
                expected = {'stage20': 20, 'stage100': 100, 'full': 1000}[args.stage]
                if data.get('samples') == expected:
                    summaries[(variant, model)] = data

    for group, items in GROUPS.items():
        path = output / f'ablation_{group}.csv'
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['variant', 'method', 'model', 'samples', 'ASR', 'AvgSim', 'DeltaASR_vs_A3', 'DeltaAvgSim_vs_A3'])
            for variant, label in items:
                for model in MODELS:
                    d = summaries.get((variant, model))
                    full = summaries.get(('A3', model))
                    writer.writerow([variant, label, model,
                                     d['samples'] if d else '',
                                     d['ASR'] if d else '',
                                     d['AvgSim'] if d else '',
                                     round(d['ASR'] - full['ASR'], 6) if d and full else '',
                                     round(d['AvgSim'] - full['AvgSim'], 6) if d and full else ''])

    lines = [f'# DHRA P0 ablation results ({args.stage})', '',
             'These scores use open-source VLM captions and a local Qwen2.5-VL-7B text judge; they are not the paper GPT-4o/Gemini protocol.',
             'The new A3 run is used for matched random crops and the first indexed image pairs. The prior 1000-image Full run remains untouched.',
             'All methods use epsilon=16/255, alpha=1/255, 300 steps and the same per-image seed.',
             'Current code uses all Transformer block CLS states; dynamic layer sampling is absent.', '',
             f'Completed evaluations: {len(summaries)}/{len(variants)*len(MODELS)}.', '']
    for group, items in GROUPS.items():
        lines += [f'## {group.title()}', '', '| Method | Qwen3B ASR / AvgSim | Qwen7B ASR / AvgSim | LLaVA1.5 ASR / AvgSim | LLaVA1.6 ASR / AvgSim |',
                  '|---|---:|---:|---:|---:|']
        for variant, label in items:
            cells = []
            for model in MODELS:
                d = summaries.get((variant, model))
                cells.append(f"{d['ASR']:.3f} / {d['AvgSim']:.3f}" if d else 'pending')
            lines.append('| ' + label + ' | ' + ' | '.join(cells) + ' |')
        lines.append('')
    lines += ['## Notes', '',
              '- Code first L2-normalizes each layer CLS vector before forming Q = ZZᵀ/L. The plan writes Q = ZZᵀ/L without stating this normalization.',
              '- Code computes squared Uhlmann fidelity via matrix square roots and clamps the result to 1.0; decoherence is max(0, C_target - C_adv)/(C_target + 1e-8).',
              '- Coherence C is the sum of absolute off-diagonal Gram entries divided by L-1. All Transformer block CLS states are included in every step.',
              '- B1 uses the planned Frobenius formula including division by L². Its initial gradient is much smaller than Base; fixed-coefficient scores alone cannot establish that Quantum Fidelity is intrinsically superior.',
              '- A1 is reused as B0 and C0; A3 is reused as C3.',
              '- A0 stage100 scores are computed from the first 100 records of its completed 1000-image ablation run; all other stage100 variants use their first 100 images.',
              '- The earlier Full Method evaluation may differ because its random crop sequence was interrupted/resumed; matched A3 is rerun here.', '']
    (output / 'ablation_summary.md').write_text('\n'.join(lines), encoding='utf-8')

    tex = ['% P0 ablation table; local Qwen7B judge, sample count given by stage',
           '\\begin{tabular}{llcccc}', '\\toprule',
           'Group & Method & Qwen3B & Qwen7B & LLaVA1.5 & LLaVA1.6 \\\\', '\\midrule']
    for group, items in GROUPS.items():
        for variant, label in items:
            cells = []
            for model in MODELS:
                d = summaries.get((variant, model))
                cells.append(f"{d['ASR']:.3f}/{d['AvgSim']:.3f}" if d else '--')
            tex.append(f"{group} & {label} & " + ' & '.join(cells) + ' \\\\')
        tex.append('\\midrule')
    tex += ['\\bottomrule', '\\end{tabular}']
    (output / 'ablation_tables.tex').write_text('\n'.join(tex), encoding='utf-8')
    print('REPORT', output, len(summaries), 'of', len(variants)*len(MODELS))


if __name__ == '__main__':
    main()
