#!/usr/bin/env python3
"""Run 20-epoch prediction-export experiments for deduplicated sensitivity."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "SRTP/HVFormer-main/HVFormer-main/scripts/train_ner_baselines.py"
OUT = ROOT / "paper_submission/results/deduplicated_sensitivity"
RAW = OUT / "raw"
LOGS = OUT / "logs"


def has_prediction(model_tag: str, seed: int) -> bool:
    for path in RAW.glob("*_test_predictions.jsonl"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if f'"seed": {seed}' not in text:
            continue
        if model_tag == "roberta_ctx" and '"bert_name": "roberta-base"' in text and '"paired_context": "auto"' in text:
            return True
        if model_tag == "bert_linear" and '"bert_name": "bert-base-uncased"' in text and '"paired_context": "none"' in text:
            return True
    return False


def run_one(name: str, args: list[str]) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.log"
    print(f"[run] {name} -> {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(args, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, text=True)
        code = proc.wait()
    if code != 0:
        raise SystemExit(f"{name} failed with exit code {code}; see {log_path}")


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    jobs = []
    for seed in [1, 2, 3, 4, 5]:
        jobs.append(
            (
                "roberta_ctx",
                seed,
                [
                    sys.executable,
                    str(TRAIN),
                    "--dataset_name",
                    "caption_ner",
                    "--model_type",
                    "bert_linear",
                    "--bert_name",
                    "roberta-base",
                    "--num_epochs",
                    "20",
                    "--batch_size",
                    "16",
                    "--lr",
                    "2e-5",
                    "--warmup_ratio",
                    "0.06",
                    "--max_seq",
                    "128",
                    "--seed",
                    str(seed),
                    "--device",
                    "mps",
                    "--log_steps",
                    "200",
                    "--paired_context",
                    "auto",
                    "--loss_on_eval_tokens_only",
                    "--export_predictions_dir",
                    str(RAW),
                ],
            )
        )
    for seed in [1, 2, 3]:
        jobs.append(
            (
                "bert_linear",
                seed,
                [
                    sys.executable,
                    str(TRAIN),
                    "--dataset_name",
                    "caption_ner",
                    "--model_type",
                    "bert_linear",
                    "--bert_name",
                    "bert-base-uncased",
                    "--num_epochs",
                    "20",
                    "--batch_size",
                    "16",
                    "--lr",
                    "3e-5",
                    "--warmup_ratio",
                    "0.06",
                    "--max_seq",
                    "80",
                    "--seed",
                    str(seed),
                    "--device",
                    "mps",
                    "--log_steps",
                    "200",
                    "--export_predictions_dir",
                    str(RAW),
                ],
            )
        )
    for model_tag, seed, args in jobs:
        if has_prediction(model_tag, seed):
            print(f"[skip] {model_tag} seed={seed} already has predictions", flush=True)
            continue
        run_one(f"{model_tag}_seed{seed}", args)
    print("[done] all requested deduplicated-sensitivity prediction exports are available", flush=True)


if __name__ == "__main__":
    main()
