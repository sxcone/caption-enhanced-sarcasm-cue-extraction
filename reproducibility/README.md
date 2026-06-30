# Reproducibility scripts

This directory contains public scripts used to audit and reproduce the cue-extraction experiments when local data are available.

It intentionally does not include dataset files, image files, result summaries, manuscript files, LaTeX sources, submitted PDFs, checkpoints, full training logs, private API keys, or raw API provider responses.

## Layout

```text
reproducibility/
└── scripts/           # dataset audits, experiment runners, and evaluation utilities
```

## Main scripts

- `dataset_stats.py`: compute split-level and label-level dataset statistics.
- `run_boundary_calibration_experiments.py`: run BC-CE and CuePrior calibration experiments.
- `run_20epoch_strong_search.sh`: run the strong encoder and paired-context search.
- `run_deduplicated_sensitivity_experiments.py`: export per-sample predictions for deduplicated-test sensitivity analysis.
- `evaluate_deduplicated_sensitivity.py`: recompute metrics under original, text-filtered, image-filtered, and deduplicated test masks.
- `calculate_annotation_reliability.py`: compute annotation agreement metrics from two reviewer sheets.
- `build_caption_quality_clip_diagnostic.py`: compute caption-quality and CLIP-alignment diagnostics.
- `run_dashscope_vlm_api_baseline.py`: run the DashScope Qwen-VL API diagnostic when `DASHSCOPE_API_KEY` is configured.
- `evaluate_qwen_vl_api_baseline.py`: evaluate parsed Qwen-VL outputs by phrase-to-token alignment.

## Notes

- API scripts require users to provide their own credentials through environment variables. No API key is stored in this repository.
- Raw Qwen/VLM outputs and result summaries are not published here.
- Large checkpoints, logs, local model caches, and manuscript build files are excluded by `.gitignore`.
