# Targeted attack evaluation

The evaluation matches each adversarial PNG to the target PNG with the same
numeric filename. It asks the selected target VLM to caption both images using
one fixed prompt, then asks GPT-4o to score the descriptions using the repository's
existing GPTScorer prompt. ASR is the fraction of scores >= 0.5; AvgSim is the
mean score. Records are saved after every API response and are reused on restart.

## API credentials

Set the required keys as environment variables in the shell that launches the
evaluation: OPENAI_API_KEY for GPT captioning or judging, GEMINI_API_KEY for
Gemini captioning, and ANTHROPIC_API_KEY for Claude captioning. Alternatively,
create an untracked api_keys.toml in this directory with keys gpt, gemini, claude.
Do not commit or print credentials.

## Commands

Check image pairing without API calls:

    python evaluate_attack.py --method DHRA --target-dir datas/target_1000 --model gpt-4o --limit 200 --dry-run

Evaluate the 200-image pilot:

    python evaluate_attack.py --method DHRA --target-dir datas/target_1000 --model gpt-4o --limit 200

Evaluate the official NIPS 1000 run after all images are generated:

    python evaluate_attack.py --method DHRA_NIPS1000 --target-dir datas/coco_val2017_1000 --model gpt-4o --limit 1000

Other supported target model names start with gemini-, claude-, gpt-, o3 or o4.
The judge defaults to gpt-4o; use --judge to change it. Run under screen for
long evaluations. Results are stored in cache/evaluation/<method>/<model>/judge_<judge>/
as records.json and summary.json.

Each target model requires two caption calls and one judge call per pair. Running
the command again resumes without repeating completed calls. A non-empty
description and numeric score are required before a pair is marked complete.
