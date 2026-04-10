# MedGemma Multilingual SRH Adapter Fine-Tuning

This repository trains, evaluates, and publishes multilingual SRH LoRA adapters
for `google/medgemma-4b-it`.

The project is designed around real dataset leaves such as `aka_gha`,
`amh_eth`, `eng_uga`, and `swa_ken`, not bare language codes alone. Each leaf
gets its own adapter so training, evaluation, and deployment stay aligned with
the underlying country-specific corpus.

## Overview

This codebase supports the full workflow:

1. Mirror multilingual SRH data from a Hugging Face dataset repo into local
   `DatasetDict.save_to_disk()` caches
2. Train one LoRA adapter per dataset leaf
3. Evaluate trained adapters on the corresponding test split
4. Publish adapters into a single Hugging Face model repo
5. Run inference locally from saved adapters or directly from Hub-hosted
   adapters

## Supported Languages

Grouped CLI selections expand to the following dataset leaves:

- `aka` -> `aka_gha`
- `amh` -> `amh_eth`
- `eng` -> `eng_eth eng_gha eng_ken eng_uga`
- `lug` -> `lug_uga`
- `swa` -> `swa_ken swa_uga`

You can also pass explicit leaves such as `eng_uga` or `swa_ken`.

## Dataset Assumptions

The dataset is expected to expose:

- `train`, `dev`, `test` splits
- `input` and `output` columns
- shard files such as `train-*`, `dev-*`, `test-*`

The loader supports both local and Hub layouts, including case variations such
as:

```text
aka/aka_gha/train-*
Aka/Aka_Gha/train-*
aka_gha/train-*
```

## Project Structure

```text
.
├── config.py
├── data_utils.py
├── prepare_data.py
├── train.py
├── evaluation.py
├── compare_models.py
├── push_adapters_to_hub.py
├── run_inference_from_hub.py
├── requirements.txt
└── README.md
```

What each file does:

- `config.py`
  Registry of supported dataset leaves, grouped language aliases, and
  per-language training settings.
- `data_utils.py`
  Shared dataset loader for Hub shards, local shard trees, and local
  `save_to_disk()` mirrors. Also contains low-resource augmentation logic.
- `prepare_data.py`
  Mirrors selected dataset leaves from Hugging Face into local `./data/<leaf>/`
  caches.
- `train.py`
  Main training entry point. Trains one adapter per leaf and can also run
  evaluation with `--eval_only`.
- `evaluation.py`
  Evaluation engine used by `train.py` for EM, token F1, ROUGE-L, and invalid
  rate reporting.
- `compare_models.py`
  Compares a baseline MedGemma checkpoint against each saved adapter on an
  evaluation split and writes both a row-level CSV and JSON reports.
- `push_adapters_to_hub.py`
  Publishes local adapters into one Hugging Face model repo.
- `run_inference_from_hub.py`
  Downloads one published adapter from Hugging Face and runs a sample prompt.

## Runtime Artifacts

After running the workflow, you will typically have:

```text
data/
├── aka_gha/
├── amh_eth/
├── eng_eth/
├── eng_gha/
├── eng_ken/
├── eng_uga/
├── lug_uga/
├── swa_ken/
└── swa_uga/

adapters/
├── adapter_aka_gha/
├── adapter_amh_eth/
├── adapter_eng_eth/
├── ...
└── eval_report.json
```

Note:

- PEFT may save named adapters in nested directories such as
  `adapter_amh_eth/amh_eth/adapter_config.json`
- The publish and inference scripts in this repo handle that layout
  automatically

## Environment Setup

Create or activate your environment, install dependencies, and export your
token:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export HF_TOKEN=hf_your_token_here
```

If you are using a managed VM with an existing environment, just activate it
and set `HF_TOKEN`.

## End-to-End Workflow

### 1. Mirror Dataset Locally

Mirror all supported languages:

```bash
python3 prepare_data.py \
  --dataset_repo AiHub4MSRH-Hash/RAW_HASH_DATASET \
  --languages aka amh eng lug swa \
  --output_root ./data
```

Mirror only selected leaves or groups:

```bash
python3 prepare_data.py \
  --dataset_repo AiHub4MSRH-Hash/RAW_HASH_DATASET \
  --languages amh eng \
  --output_root ./data
```

### 2. Train One Adapter as a Smoke Test

Example:

```bash
python3 train.py \
  --data_root ./data \
  --languages amh \
  --output_root ./adapters \
  --max_eval_samples 50
```

This:

- loads `amh_eth`
- applies low-resource augmentation if configured
- trains the adapter
- evaluates it
- writes results to `./adapters/eval_report.json`

### 3. Safest Production Workflow: Train One Leaf at a Time

This is the recommended approach on storage-constrained VMs.

Recommended order:

1. `eng_eth`
2. `eng_gha`
3. `eng_ken`
4. `eng_uga`
5. `aka_gha`
6. `amh_eth`
7. `lug_uga`
8. `swa_ken`
9. `swa_uga`

Example pattern:

```bash
python3 train.py \
  --data_root ./data \
  --languages eng_eth \
  --output_root ./adapters \
  --max_eval_samples 100
```

After each run, remove checkpoint directories to control disk usage:

```bash
find ./adapters -type d -name "checkpoint-*" -prune -exec rm -rf {} +
```

Useful disk checks:

```bash
df -h
du -sh ./adapters
du -sh ./adapters/*
```

### 4. Evaluate All Trained Adapters

After all leaf-by-leaf training is complete:

```bash
python3 train.py \
  --eval_only \
  --data_root ./data \
  --languages aka amh eng lug swa \
  --output_root ./adapters \
  --max_eval_samples 200
```

This updates:

- `./adapters/eval_report.json`

## Training Commands

### Train From Local Mirrors

All languages:

```bash
python3 train.py \
  --data_root ./data \
  --languages aka amh eng lug swa \
  --output_root ./adapters
```

Specific leaves:

```bash
python3 train.py \
  --data_root ./data \
  --languages eng_uga swa_ken \
  --output_root ./adapters
```

### Train Directly From a Hugging Face Dataset Repo

```bash
python3 train.py \
  --dataset_repo AiHub4MSRH-Hash/RAW_HASH_DATASET \
  --languages aka amh eng lug swa \
  --output_root ./adapters
```

### Evaluation Only

From local data:

```bash
python3 train.py \
  --eval_only \
  --data_root ./data \
  --languages aka amh eng lug swa \
  --output_root ./adapters \
  --max_eval_samples 200
```

From Hub data:

```bash
python3 train.py \
  --eval_only \
  --dataset_repo AiHub4MSRH-Hash/RAW_HASH_DATASET \
  --languages aka amh eng lug swa \
  --output_root ./adapters \
  --max_eval_samples 200
```

## Push Adapters to Hugging Face

### Push All Local Adapters

```bash
python3 push_adapters_to_hub.py \
  --repo_id your-org/hashie-srh-adapters \
  --output_root ./adapters \
  --private
```

### Push Only Selected Languages

```bash
python3 push_adapters_to_hub.py \
  --repo_id your-org/hashie-srh-adapters \
  --output_root ./adapters \
  --languages aka amh eng lug swa \
  --private
```

This uploads:

- `adapters/<dataset_id>/...`
- `adapters/manifest.json`
- `README.md`
- `reports/eval_report.json` if present

Expected Hub layout:

```text
README.md
adapters/
├── manifest.json
├── aka_gha/
├── amh_eth/
├── eng_eth/
├── eng_gha/
├── eng_ken/
├── eng_uga/
├── lug_uga/
├── swa_ken/
└── swa_uga/
reports/
└── eval_report.json
```

## Inference

### Baseline vs Adapter Comparison

Run a baseline checkpoint against the same evaluation split as each adapter and
save both row-level generations and summary reports:

```bash
python3 compare_models.py \
  --data_root ./data \
  --languages eng_uga swa_ken \
  --output_root ./adapters \
  --baseline_model /path/to/your/original-finetuned-medgemma \
  --max_eval_samples 100 \
  --load_in_4bit
```

This writes:

- `./adapters/adapter_baseline_comparison.csv`
- `./adapters/baseline_eval_report.json`
- `./adapters/adapter_comparison_report.json`

The CSV includes the dataset leaf, original question, reference answer,
baseline model prediction, and adapter prediction so you can inspect exactly
where the adapters helped or regressed.

### Local Inference From a Saved Adapter

There is no dedicated CLI script for local inference in this repo, but you can
run it with a short Python snippet using the helper in `train.py`.

Example:

```bash
python3 - <<'PY'
from pathlib import Path
from train import load_adapter_for_inference
from config import TrainingConfig
import torch

cfg = TrainingConfig()
model, processor = load_adapter_for_inference(
    language="amh_eth",
    cfg=cfg,
    output_root=Path("adapters"),
)

messages = [
    {
        "role": "system",
        "content": (
            "You are Hashie, a multilingual medical assistant with expertise in "
            "sexual and reproductive health. Answer in Amharic."
        ),
    },
    {"role": "user", "content": "What are common symptoms of an STI?"},
]

text = processor.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
inputs = processor(text=text, return_tensors="pt")
if torch.cuda.is_available():
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
        temperature=None,
        top_p=None,
    )

prompt_length = inputs["input_ids"].shape[1]
new_tokens = output_ids[0][prompt_length:]
response = processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
print(response)
PY
```

### Hub Inference

Run inference against one published adapter:

```bash
python3 run_inference_from_hub.py \
  --adapter_repo your-org/hashie-srh-adapters \
  --adapter_name amh_eth \
  --prompt "What are common symptoms of an STI?"
```

Optional low-memory loading:

```bash
python3 run_inference_from_hub.py \
  --adapter_repo your-org/hashie-srh-adapters \
  --adapter_name swa_ken \
  --prompt "Shida ya Ukimwi ni nini?" \
  --load_in_4bit
```

You can also override the base model:

```bash
python3 run_inference_from_hub.py \
  --adapter_repo your-org/hashie-srh-adapters \
  --adapter_name eng_uga \
  --base_model google/medgemma-4b-it \
  --prompt "When should someone seek urgent care during pregnancy?"
```

## Practical Operating Notes

- Train one leaf at a time if disk space is limited
- Remove `checkpoint-*` folders after successful runs if you do not need resume
  training
- `./adapters/eval_report.json` is overwritten when evaluation runs again
- Hub publish and Hub inference now support both flat and nested PEFT adapter
  layouts
- Exact Match is usually very strict for generative SRH answers; use token F1
  and ROUGE-L alongside manual spot checks

## Common Commands Cheat Sheet

Mirror data:

```bash
python3 prepare_data.py \
  --dataset_repo AiHub4MSRH-Hash/RAW_HASH_DATASET \
  --languages aka amh eng lug swa \
  --output_root ./data
```

Train one language:

```bash
python3 train.py \
  --data_root ./data \
  --languages amh_eth \
  --output_root ./adapters \
  --max_eval_samples 100
```

Evaluate all:

```bash
python3 train.py \
  --eval_only \
  --data_root ./data \
  --languages aka amh eng lug swa \
  --output_root ./adapters \
  --max_eval_samples 200
```

Push all adapters:

```bash
python3 push_adapters_to_hub.py \
  --repo_id your-org/hashie-srh-adapters \
  --output_root ./adapters \
  --private
```

Run Hub inference:

```bash
python3 run_inference_from_hub.py \
  --adapter_repo your-org/hashie-srh-adapters \
  --adapter_name amh_eth \
  --prompt "What are common symptoms of an STI?"
```

Compare a baseline model versus local adapters:

```bash
python3 compare_models.py \
  --data_root ./data \
  --languages eng \
  --output_root ./adapters \
  --baseline_model /path/to/your/original-finetuned-medgemma \
  --max_eval_samples 100 \
  --load_in_4bit
```
