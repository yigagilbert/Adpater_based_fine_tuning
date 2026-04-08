# MedGemma Multilingual SRH Adapter Training

This codebase is now aligned to a Hugging Face-hosted multilingual SRH dataset that uses:

- ISO 639-2/3 language codes: `aka`, `eng`, `lug`, `swa`
- Dataset leaves with country suffixes: `aka_gha`, `eng_uga`, `swa_ken`, etc.
- `dev` instead of `validation`
- shard globs such as `train-*`, `dev-*`, and `test-*`
- two required columns per split: `input` and `output`

## Dataset Layout

```text
aka/
└── aka_gha/
    ├── train-*
    ├── dev-*
    └── test-*
eng/
├── eng_eth/
├── eng_gha/
├── eng_ken/
└── eng_uga/
lug/
└── lug_uga/
swa/
├── swa_ken/
└── swa_uga/
```

The training registry treats each leaf like `eng_uga` or `swa_ken` as a first-class dataset target. You can still pass grouped selections such as `eng` or `swa` on the CLI, and they expand to all matching leaves.

## What Changed

- `config.py` now registers the real dataset leaves instead of old ISO 639-1 codes.
- `data_utils.py` now loads from:
  - a Hugging Face repo with shard globs
  - a local mirrored shard tree
  - a local `save_to_disk()` mirror
- split handling is canonicalized to `train` / `dev` / `test`, while still tolerating legacy `validation` and `val`
- `train.py` and `evaluation.py` now use the shared dataset loader instead of assuming `load_from_disk(data/<lang_code>)`
- `prepare_data.py` is now a Hub mirroring utility, not a CSV/JSONL converter

## Train From Hugging Face

```bash
HF_TOKEN=hf_xxx python3 train.py \
  --dataset_repo your-org/your-srh-dataset \
  --languages aka eng lug swa \
  --output_root ./adapters
```

Grouped selections expand as follows:

- `aka` -> `aka_gha`
- `eng` -> `eng_eth eng_gha eng_ken eng_uga`
- `lug` -> `lug_uga`
- `swa` -> `swa_ken swa_uga`

You can also target specific leaves:

```bash
HF_TOKEN=hf_xxx python3 train.py \
  --dataset_repo your-org/your-srh-dataset \
  --languages eng_uga swa_ken \
  --output_root ./adapters
```

## Mirror Hub Data Locally

```bash
python3 prepare_data.py \
  --dataset_repo your-org/your-srh-dataset \
  --languages eng swa \
  --output_root ./data
```

This writes local `DatasetDict.save_to_disk()` mirrors like:

```text
data/
├── eng_eth/
├── eng_gha/
├── eng_ken/
├── eng_uga/
├── swa_ken/
└── swa_uga/
```

After mirroring, training can run without `--dataset_repo`:

```bash
HF_TOKEN=hf_xxx python3 train.py \
  --data_root ./data \
  --languages eng swa \
  --output_root ./adapters
```

## Evaluate Only

```bash
python3 train.py \
  --eval_only \
  --dataset_repo your-org/your-srh-dataset \
  --languages aka eng lug swa \
  --output_root ./adapters
```

## Publish Adapters To The Hub

After training, you can publish all local `adapter_*` folders into one Hugging
Face model repo.

```bash
HF_TOKEN=hf_xxx python3 push_adapters_to_hub.py \
  --repo_id your-org/hashie-srh-adapters \
  --output_root ./adapters
```

To publish only selected leaves:

```bash
HF_TOKEN=hf_xxx python3 push_adapters_to_hub.py \
  --repo_id your-org/hashie-srh-adapters \
  --output_root ./adapters \
  --languages eng_uga swa_ken
```

The Hub repo layout will look like:

```text
README.md
adapters/
├── manifest.json
├── eng_uga/
├── swa_ken/
└── ...
reports/
└── eval_report.json
```

## Run Inference From The Hub

The sample script downloads only the requested adapter subtree from the Hub repo,
reads `adapter_meta.json` to recover the base model, and runs one prompt.

```bash
HF_TOKEN=hf_xxx python3 run_inference_from_hub.py \
  --adapter_repo your-org/hashie-srh-adapters \
  --adapter_name eng_uga \
  --prompt "What are common symptoms of an STI?"
```

Optional low-memory loading:

```bash
HF_TOKEN=hf_xxx python3 run_inference_from_hub.py \
  --adapter_repo your-org/hashie-srh-adapters \
  --adapter_name swa_ken \
  --prompt "How can HIV transmission be reduced?" \
  --load_in_4bit
```

## Notes

- Adapters are saved per dataset leaf, for example `adapter_eng_uga/`.
- Evaluation reads the `test-*` shards for the same leaf.
- Low-resource augmentation still exists, but donor transfer is now configured between real dataset leaves such as `swa_uga -> swa_ken` and `lug_uga -> eng_uga`.
