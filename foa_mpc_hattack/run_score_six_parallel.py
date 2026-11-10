"""Score the six complete GPT/Gemini caption sets independently and resumably."""
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MODELS = ("gpt-4o-2024-05-13", "gemini-3.6-flash")
VARIANTS = ("MPC_noMI_LDI", "FOA_MPC_LDI", "FOA_MPC_Hattack_LDI_C01")
OUTPUT = "cache/evaluation/foa_mpc_hattack_302_judge_gpt4o_2024_05_13"


def run(model, variant):
    cmd = [sys.executable, "-m", "foa_mpc_hattack.eval_302", "--models", model,
           "--variants", variant, "--score-only", "--judge-model", "gpt-4o-2024-05-13",
           "--output", OUTPUT]
    log = Path("/root/autodl-tmp") / f"score_{model}_{variant}.log"
    env = dict(os.environ, HTTPS_PROXY="http://127.0.0.1:10810",
               HTTP_PROXY="http://127.0.0.1:10810")
    with log.open("a", encoding="utf-8") as stream:
        for attempt in range(60):
            result = subprocess.run(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode == 0:
                print("COMPLETE", model, variant, flush=True)
                return
            print("RETRY", model, variant, attempt + 1, flush=True)
            time.sleep(60)
    raise RuntimeError(f"Failed after retries: {model} {variant}")


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(lambda pair: run(*pair),
                      ((model, variant) for model in MODELS for variant in VARIANTS)))
