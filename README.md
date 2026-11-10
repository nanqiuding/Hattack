# Hattack

Research code for targeted transferable attacks on vision-language models, including DHRA/Hattack, M-Attack, AdvDiffVLM adaptations, and FOA+MPC experiments. The attack source images are from NIPS2017 and the paired targets are from COCO val2017. Dataset images, model checkpoints, generated attacks, caption caches, and API credentials are not included.

## Repository layout

- `methods/`: original attack implementations and DHRA components.
- `gen_adv.py`, `config.toml`: main attack entry point and configuration.
- `foa_mpc_hattack/`: matched FOA/MPC variants, layer-fidelity experiments, local open-model evaluation, and resumable 302.AI caption/scoring scripts.
- `run_mattack_1000.py`, `run_mattack_100.py`: M-Attack generation runners.
- `caption_open_models.py`, `score_open_captions.py`: open-model captioning and local judging.
- `EVALUATION.md`, `README_FOA_MPC_HATTACK.md`: evaluation protocol and experiment details.

The FOA/MPC experiment implementation and its variant definitions are documented in [`README_FOA_MPC_HATTACK.md`](README_FOA_MPC_HATTACK.md). Its vendored third-party source snapshots are identified there by upstream repository and commit. Large pretrained weights must be downloaded separately.

## Data and runs

Place the paired PNG files in `datas/nips2017_1000/` and `datas/coco_val2017_1000/`. The sorted first `N` filenames are paired by the runners. Review `config.toml` and the individual `--help` output for attack settings. For the three-surrogate, no-momentum MPC variant used in the 100-pair comparison:

```bash
python -m foa_mpc_hattack.run_attack \
  --variant E4LDINoMIFOA --num-samples 100 \
  --out-dir cache/foa_mpc_hattack/stage100
```

The API evaluator reads `AI302_API_KEY` or `~/.config/hattack/302_key` and stores captions and scores under `cache/evaluation/`. Keep credentials outside the repository. The local open-model evaluator is `python -m foa_mpc_hattack.run_eval --help`; the API evaluator is `python -m foa_mpc_hattack.eval_302 --help`.
