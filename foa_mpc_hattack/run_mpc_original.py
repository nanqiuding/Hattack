"""Launch the MPC authors' attack script with fixed paired input files.

Only CLI sample-count, seed, and step controls were added to the vendored copy.
The authors' feature concatenation, text mixing, InfoNCE and MI-FGSM remain.
"""
import os
import subprocess
import sys
from pathlib import Path


def run_original(args, pairs):
    vendor = Path(__file__).resolve().parent / "vendor" / "mpc_official"
    data = args.out_dir / f"{args.variant}_imagefolder"
    for kind, paths in (("source", [s for s, _ in pairs]),
                        ("target", [t for _, t in pairs])):
        folder = data / kind / "images"
        folder.mkdir(parents=True, exist_ok=True)
        for path in paths:
            link = folder / path.name
            if not link.exists():
                link.symlink_to(path.resolve())
    cmd = [sys.executable, "generate_adversarial_examples_MPCAttack.py",
           "--cle_data_path", str(data / "source"),
           "--tgt_data_path", str(data / "target"),
           "--output", str(args.out_dir / f"{args.variant}_mpc_official"),
           "--device", args.device,
           "--num_samples", str(args.num_samples),
           "--seed", str(args.seed),
           "--steps", str(args.steps),
           "--epsilon", str(args.epsilon),
           "--alpha", str(args.alpha)]
    if args.variant in ("E4NoMI", "E4NoMIFOA", "E4LDINoMIFOA"):
        cmd.append("--no_mi")
    if args.variant in ("E4NoMIFOA", "E4LDINoMIFOA"):
        cmd.append("--foa_init_crop")
    if args.variant in ("E4LDI", "E4LDINoMIFOA"):
        cmd.extend(("--clip_backbone", "Laion"))
    print("MPC_OFFICIAL_COMMAND", " ".join(cmd), flush=True)
    env = dict(os.environ)
    # The official source imports kmeans_pytorch as a top-level package.
    # Hattack already carries that dependency in its repository root.
    env["PYTHONPATH"] = os.pathsep.join((str(vendor), str(args.data_root.parent)))
    subprocess.run(cmd, cwd=vendor, env=env, check=True)
