# Caption-enhanced Sarcasm Cue Extraction

This repository contains the training and evaluation code for multimodal sarcasm cue extraction.

The task is formulated as BIO sequence labelling over two channels:

- `text_ner`: cue extraction from the original post text.
- `caption_ner`: cue extraction from generated image captions.

The repository intentionally includes code only. Dataset files, image files, manuscript files, paper figures, result tables, checkpoints, logs and temporary build artifacts are not included.

## Repository layout

```text
.
├── SRTP/HVFormer-main/HVFormer-main/       # training and model code
└── reproducibility/                        # public audit/evaluation scripts
```

## Dataset

The dataset and image files are not distributed in this code repository. The training code expects local BIO files in a CoNLL-like format:

```text
IMGID:1
token<TAB>B-TOPI
token<TAB>O

IMGID:2
...
```

Label meanings:

- `TENT`: textual sarcasm-related entity.
- `TOPI`: textual opinion cue.
- `MENT`: visual/caption-side entity or anchor.
- `MOPI`: visual/caption-side opinion cue.

## Environment

Create an environment and install dependencies:

```bash
cd SRTP/HVFormer-main/HVFormer-main
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The code expects local Hugging Face model files for `bert-base-uncased` and `openai/clip-vit-base-patch32` if offline mode is enabled in `run.py`.

## Example training commands

Run caption-side HVFormer-style cue extraction:

```bash
cd SRTP/HVFormer-main/HVFormer-main
python run.py \
  --model_name bert-vit-inter-ner \
  --dataset_name caption_ner \
  --bert_name bert-base-uncased \
  --vit_name openai/clip-vit-base-patch32 \
  --device mps \
  --num_epochs 20 \
  --seed 1
```

## Reproducibility utilities

Additional experiment and audit scripts are provided under `reproducibility/`.
These scripts cover dataset statistics, annotation agreement, boundary-calibrated cue extraction, cue-prior calibration, strong encoder search, deduplicated-test sensitivity analysis, caption-quality diagnostics and VLM prompt diagnostics.

The repository does not include result-summary files, raw outputs, paper source files, submitted PDFs, checkpoints, full logs, private API keys or raw API provider responses.

Run text-side cue extraction:

```bash
cd SRTP/HVFormer-main/HVFormer-main
python run.py \
  --model_name bert-vit-inter-ner \
  --dataset_name text_ner \
  --bert_name bert-base-uncased \
  --vit_name openai/clip-vit-base-patch32 \
  --device mps \
  --num_epochs 20 \
  --seed 1
```

Run BERT/RoBERTa sequence-labelling baselines:

```bash
cd SRTP/HVFormer-main/HVFormer-main
python scripts/train_ner_baselines.py \
  --dataset_name caption_ner \
  --encoder_name roberta-base \
  --head linear \
  --paired_context auto \
  --num_epochs 20 \
  --seed 1 \
  --device mps
```

Use `--device cuda` on CUDA machines or `--device cpu` when no accelerator is available.

## Notes

- Checkpoints, logs and cached model files are excluded from this repository.
- Dataset and image files are excluded from this repository.
- Users who run the scripts should provide their own local data paths that follow the expected BIO format.
