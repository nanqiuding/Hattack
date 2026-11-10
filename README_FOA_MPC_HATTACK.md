# FOA + MPC surrogates + Hattack fidelity

This directory is isolated from the existing Hattack methods and generated images.
The experiment uses the first `N` sorted files in `datas/nips2017_1000` and
`datas/coco_val2017_1000`; `pair_manifest.csv` records the exact pairing.

## Provenance and variants

The vendored FOA surrogate and loss classes come from
`jiaxiaojunQAQ/FOA-Attack` commit `6aa9ad463a1e812b7f77ff2e5c4b1712e7c78c4b`.
The MPC runner and surrogate source come from `LiYuanBoJNU/MPCAttack` commit
`7b195a5bcbaeaf1e8534d101199810a2782caad3`, with only the CLI controls
for sample count, step count, and seed added to its attack script. Those
snapshots are included under `vendor/` for auditability.
The E4 copy also appends per-image runtime, peak VRAM, and perturbation norms
to a JSONL ledger; this does not change its loss or update rule. The vendored
MPC package initializer imports only the extractors needed by E4, avoiding
unrelated Qwen/DINOv3 optional dependencies.

| Variant | Surrogates | Active objective |
| --- | --- | --- |
| E0 | FOA CLIP B/16, B/32, LAION G/14 | Official FOA clustered OT and dynamic weighting; its global term is singleton OT on visual hidden CLS. |
| E1 | E0 plus DINOv2-base and InternVL3-1B | Same official FOA objective on the three CLIPs plus the mean within-model target cosine of DINOv2 and InternVL3. Their local OT is **off**, because equivalence to FOA patch OT is unverified. |
| E2 | Identical E1 surrogates and terms | E1 plus `-lambda_coh * mean(1 - squared_Uhlmann_fidelity)` over the three CLIPs only. No decoherence. |
| E4 | Official MPC CLIPs, DINOv2 and InternVL3 | Official text/image mixing, feature concatenation, InfoNCE and MI-FGSM code. Its objective is independent of FOA. |

FOA and MPC use the exact same LAION model ID in the checked official source:
`laion/CLIP-ViT-G-14-laion2B-s12B-b42K`. E0/E1/E2 use the official FOA
cluster count 10, local coefficient 0.2, Sinkhorn temperature 0.1, and
previous-score-ratio softmax weights. FOA's code maximizes its score with
`delta += alpha * sign(grad)`; the runner preserves this direction and the
FOA 8-bit perturbation units. The official FOA source computes its target
features from a fresh target crop each step, so these are not cached across
steps. E0/E1/E2 use the same crops, iterations, seed, and update rule.

Differences to a bitwise execution of the FOA authors' top-level script:
the runner uses per-image `seed + index` to make matched variants comparable,
does not initialize WandB, writes an explicit pair manifest, and clips the
final saved PNG to the valid pixel range. Its official FOA extractor and loss
implementations are unchanged. Thus E0 is a *matched FOA implementation*
and any comparison with published FOA scores requires protocol disclosure.

MPC's official script generates source and target descriptions with InternVL3
before attack. E4 therefore takes longer and requires the full multimodal
checkpoint. Its input `ImageFolder` views are symlinks to the same manifest
pairs. The output PNGs are not used as target data for any other variant.

## Data, weights, environment

Run from `/root/Hattack`. The data disk is nearly full from victim-model
checkpoints, so the new surrogates use the system cache. B/16 and B/32 are
linked to their already downloaded files on the data disk:

```bash
export HF_HOME=/root/.cache/huggingface
python download_foa_mpc_models.py
```

Required public model IDs are OpenAI CLIP B/16 and B/32, LAION CLIP G/14,
`facebook/dinov2-base`, and `OpenGVLab/InternVL3-1B-hf`. The large open-source
captioning victims are separate and live under `/root/autodl-tmp/vlm_models`.
The remote environment initially has PyTorch 2.8.0 and Transformers 4.57.6.
Do not silently remove a missing surrogate: the corresponding variant must
fail with a visible checkpoint error. A single RTX 4090 has 24 GB VRAM;
E1/E2/E4 may require further memory engineering, which must be recorded
before results are claimed.

## Commands

```bash
python -m unittest discover -s foa_mpc_hattack/tests -v

# Fast functional check; a CLIP-only E0 check is not a full E0 result.
python -m foa_mpc_hattack.run_attack --variant E0 --num-samples 1 \
  --steps 2 --clip-backbones B16

# Four full-surrogate pilot variants, same first five pairs.
for v in E0 E1 E2 E4; do
  python -m foa_mpc_hattack.run_attack --variant "$v" \
    --num-samples 5 --seed 2023 --steps 300 --epsilon 16 --alpha 1
done

# After pilot and validation, run matched 100 and then 1000 pairs.
python -m foa_mpc_hattack.run_attack --variant E1 --num-samples 100
python -m foa_mpc_hattack.run_attack --variant E2 --num-samples 100 \
  --lambda-coh 0.1
python -m foa_mpc_hattack.run_attack --variant E0 --num-samples 100
python -m foa_mpc_hattack.run_attack --variant E4 --num-samples 100

# Same local caption/judge protocol used for earlier Hattack results.
python -m foa_mpc_hattack.run_eval --variant E1 --num-samples 100
```

`--num-samples`, `--seed`, `--data-root`, `--weights-root`, and `--out-dir`
are common CLI controls. Attack outputs, `*_samples.jsonl`, configuration
JSON, and pair manifest are under the chosen output directory. The FOA
previous-score vector is saved per image and restored on resume, so the
original dynamic weighting state is preserved across interrupted runs.
Evaluation creates symlinks under `cache/adv_images/FOA_MPC_E*` and invokes the existing
`caption_open_models.py` and `score_open_captions.py`. The resulting ASR and
AvgSim use the **local Qwen2.5-VL-7B text-only judge**, not the papers'
GPT-4o/Gemini judging protocol. Each victim has a per-sample score CSV.

## Fidelity regularization and interpretation

For a model's `[B,L,D]` layer-CLS tensor, each layer vector is normalized,
then `Q=ZZ^T/L`. An `eps I` term is added and trace-normalized; the target
square root is computed without gradients, and the inner PSD eigenvalues
are used directly for `tr(sqrt(A))` so gradients do not pass through inner
eigenvectors. `lambda_coh=0.1` is an **untuned pilot value**, not a selected
optimal value. Hyperparameter selection must use pairs disjoint from the
reported test set. If E2 does not improve upon E1, report that result.
