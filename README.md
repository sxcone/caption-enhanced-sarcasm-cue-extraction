# Caption-enhanced Sarcasm Cue Extraction

This repository contains the training code and dataset files for multimodal sarcasm cue extraction.

The task is formulated as BIO sequence labelling over two channels:

- `text_ner`: cue extraction from the original post text.
- `caption_ner`: cue extraction from generated image captions.

The repository intentionally includes only code and training data. Manuscript files, paper figures, result tables, checkpoints, logs and temporary build artifacts are not included.

## Repository layout

```text
.
├── caption和text/                         # BIO text/caption annotation files
├── part1_img_580/                         # train image split
├── part2_img_364/                         # validation image split
├── part3_img_420/                         # additional image split retained for completeness
├── part5_img_394/                         # test image split
├── SRTP/HVFormer-main/HVFormer-main/       # training and model code
└── reproducibility/                        # public audit/evaluation scripts and small summaries
```

## Dataset

The released annotation files use a CoNLL-like format:

```text
IMGID:1
token<TAB>B-TOPI
token<TAB>O

IMGID:2
...
```

Splits used by the training code:

| Channel | Train | Dev | Test |
|---|---|---|---|
| `text_ner` | `caption和text/text_part1_1000_580.txt` | `caption和text/text_part2_650_364.txt` | `caption和text/text_3545_part5_595_394.txt` |
| `caption_ner` | `caption和text/caption_3545_part1_1000_580.txt` | `caption和text/caption_part2_650_364.txt` | `caption和text/caption_3545_part5_595_394.txt` |

The split sizes are 1,230 training samples, 263 validation samples and 265 test samples.

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

The `reproducibility/results_summary/` directory contains compact public summaries used to document the reported experiments.
It does not contain paper source files, submitted PDFs, checkpoints, full logs, private API keys or raw API provider responses.

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
- The image files are provided for research reproduction with the released annotations.
- If redistribution restrictions apply to a downstream use case, keep the annotations and replace the image files with source identifiers or locally downloaded copies.
