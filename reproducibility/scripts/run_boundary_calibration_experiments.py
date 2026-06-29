#!/usr/bin/env python3
"""Run Boundary-Calibrated Cue Extractor experiments for the paper."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


PAPER_DIR = Path(__file__).resolve().parents[1]
ROOT = PAPER_DIR.parent
CODE_DIR = ROOT / "SRTP" / "HVFormer-main" / "HVFormer-main"
SRC_REPORT_DIR = CODE_DIR / "reports" / "baseline_results"
RAW_DIR = PAPER_DIR / "results" / "boundary_calibration" / "raw"
TABLE_DIR = PAPER_DIR / "tables"
RESULT_TABLE_DIR = PAPER_DIR / "results" / "tables"
LOG_PATH = PAPER_DIR / "results" / "boundary_calibration" / "experiment_log.md"
DEFAULT_CONDA_PYTHON = Path("/Users/sxc/miniforge3/bin/python")
PYTHON = os.environ.get("PAPER_PYTHON") or (
    str(DEFAULT_CONDA_PYTHON)
    if DEFAULT_CONDA_PYTHON.exists()
    else (shutil.which("python3") or shutil.which("python") or "python3")
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["caption_ner", "text_ner"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--boundary_weights", nargs="+", type=float, default=[0.1, 0.2, 0.4])
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_seq", type=int, default=80)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--smoke", action="store_true", help="Run a 1-epoch caption-only sanity check.")
    parser.add_argument("--reuse_existing", action="store_true", help="Only summarize copied JSON files.")
    return parser.parse_args()


def metric_block(data: dict, key: str) -> dict:
    metrics = data.get(key, {})
    return {
        "f1": float(metrics.get("f1", 0.0)),
        "precision": float(metrics.get("precision", 0.0)),
        "recall": float(metrics.get("recall", 0.0)),
    }


def build_command(
    dataset: str,
    seed: int,
    num_epochs: int,
    batch_size: int,
    max_seq: int,
    device: str,
    experiment_name: str,
    *,
    use_boundary: bool = False,
    boundary_weight: float = 0.0,
    tune_decode: bool = False,
) -> list[str]:
    command = [
        PYTHON,
        "run.py",
        "--do_train",
        "--disable_model_save",
        "--model_name",
        "bert-vit-inter-ner",
        "--dataset_name",
        dataset,
        "--experiment_name",
        experiment_name,
        "--num_epochs",
        str(num_epochs),
        "--batch_size",
        str(batch_size),
        "--max_seq",
        str(max_seq),
        "--device",
        device,
        "--seed",
        str(seed),
        "--log_steps",
        "20",
        "--ner_metric",
        "token_collapsed",
        "--ce_on_all_tokens",
        "--focal_gamma",
        "2.0",
        "--collapse_bio_labels",
        "--decode_with_argmax",
        "--crf_loss_weight",
        "0.0",
        "--o_loss_weight",
        "0.2",
        "--class_weight_power",
        "0.7",
        "--lr",
        "3e-5",
        "--warmup_ratio",
        "0.06",
        "--notes",
        "Boundary-Calibrated Cue Extractor ablation",
    ]
    if use_boundary:
        command.extend(["--use_span_boundary_head", "--boundary_loss_weight", str(boundary_weight)])
    if tune_decode:
        command.extend([
            "--tune_decode_o_bias",
            "--decode_o_bias_min",
            "-2.0",
            "--decode_o_bias_max",
            "2.0",
            "--decode_o_bias_step",
            "0.1",
        ])
    return command


def latest_existing_result(spec: dict) -> Path | None:
    patterns = [
        RAW_DIR.glob(f"{spec['experiment_name']}_{spec['dataset']}_*.json"),
        SRC_REPORT_DIR.glob(f"{spec['experiment_name']}_{spec['dataset']}_*.json"),
    ]
    candidates = []
    for matches in patterns:
        candidates.extend(matches)
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: path.stat().st_mtime)[-1]


def run_one(spec: dict) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    existing = latest_existing_result(spec)
    if existing is not None:
        source = existing
        print(f"Reusing {source.name}")
    else:
        source = None

    before = {p.resolve() for p in SRC_REPORT_DIR.glob(f"{spec['experiment_name']}_{spec['dataset']}_*.json")}
    if source is None and not spec["reuse_existing"]:
        subprocess.run(spec["command"], cwd=CODE_DIR, check=True)
        candidates = sorted(
            [p for p in SRC_REPORT_DIR.glob(f"{spec['experiment_name']}_{spec['dataset']}_*.json") if p.resolve() not in before],
            key=lambda path: path.stat().st_mtime,
        )
        if not candidates:
            candidates = sorted(
                SRC_REPORT_DIR.glob(f"{spec['experiment_name']}_{spec['dataset']}_*.json"),
                key=lambda path: path.stat().st_mtime,
            )
        if not candidates:
            raise FileNotFoundError(f"No result JSON found for {spec['experiment_name']} on {spec['dataset']}")
        source = candidates[-1]

    data = json.loads(source.read_text(encoding="utf-8"))
    data.update({
        "paper_method_key": spec["key"],
        "paper_method_label": spec["label"],
        "paper_seed": spec["seed"],
        "paper_boundary_weight": spec["boundary_weight"],
        "paper_decode_calibration": spec["tune_decode"],
        "paper_source_json": str(source),
    })
    target = RAW_DIR / source.name
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def make_specs(args: argparse.Namespace) -> list[dict]:
    datasets = ["caption_ner"] if args.smoke else args.datasets
    seeds = [args.seeds[0]] if args.smoke else args.seeds
    num_epochs = 1 if args.smoke else args.num_epochs
    weights = [args.boundary_weights[0]] if args.smoke else args.boundary_weights

    specs = []
    for dataset in datasets:
        for seed in seeds:
            common = {
                "dataset": dataset,
                "seed": seed,
                "num_epochs": num_epochs,
                "batch_size": args.batch_size,
                "max_seq": args.max_seq,
                "device": args.device,
                "reuse_existing": args.reuse_existing,
            }
            variants = [
                ("hvformer_text_only", "HVFormer Text-Only", False, 0.0, False),
                ("hvformer_text_only_calibrated", "HVFormer Text-Only + Calib.", False, 0.0, True),
            ]
            for weight in weights:
                weight_tag = str(weight).replace(".", "p")
                variants.extend([
                    (f"bcce_w{weight_tag}", f"BC-CE $\\lambda={weight:g}$", True, weight, False),
                    (f"bcce_w{weight_tag}_calibrated", f"BC-CE $\\lambda={weight:g}$ + Calib.", True, weight, True),
                ])
            for key, label, use_boundary, boundary_weight, tune_decode in variants:
                experiment_name = f"bcce_{key}_seed{seed}"
                specs.append({
                    **common,
                    "key": key,
                    "label": label,
                    "use_boundary": use_boundary,
                    "boundary_weight": boundary_weight,
                    "tune_decode": tune_decode,
                    "experiment_name": experiment_name,
                    "command": build_command(
                        dataset,
                        seed,
                        num_epochs,
                        args.batch_size,
                        args.max_seq,
                        args.device,
                        experiment_name,
                        use_boundary=use_boundary,
                        boundary_weight=boundary_weight,
                        tune_decode=tune_decode,
                    ),
                })
    return specs


def summarize(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "dataset": data.get("dataset_name"),
            "method_key": data.get("paper_method_key"),
            "method": data.get("paper_method_label"),
            "seed": data.get("paper_seed"),
            "boundary_weight": data.get("paper_boundary_weight"),
            "decode_calibration": bool(data.get("paper_decode_calibration")),
            "best_decode_o_bias": float(data.get("best_decode_o_bias", 0.0)),
            "best_dev_epoch": data.get("best_dev_epoch"),
            "best_dev": metric_block(data, "best_dev_metrics"),
            "final_test": metric_block(data, "final_test_metrics"),
            "json": str(path),
        })
    return sorted(rows, key=lambda row: (row["dataset"], row["seed"], row["method_key"]))


def write_tex(rows: list[dict]) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "Dataset & Method & Seed & Dev F1 & Test P & Test R & Test F1 \\\\",
        "\\midrule",
    ]
    for row in rows:
        dataset = str(row["dataset"]).replace("_", "\\_")
        method = row["method"]
        final = row["final_test"]
        best_dev = row["best_dev"]
        lines.append(
            f"{dataset} & {method} & {row['seed']} & "
            f"{best_dev['f1']:.4f} & {final['precision']:.4f} & "
            f"{final['recall']:.4f} & {final['f1']:.4f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    tex = "\n".join(lines)
    for out_dir in (TABLE_DIR, RESULT_TABLE_DIR):
        (out_dir / "boundary_calibration_results.tex").write_text(tex, encoding="utf-8")


def write_log(rows: list[dict], args: argparse.Namespace) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Boundary-Calibrated Cue Extractor Experiments",
        "",
        f"- Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"- Python: `{PYTHON}`",
        f"- Device: `{args.device}`",
        f"- Epochs: `{1 if args.smoke else args.num_epochs}`",
        f"- Smoke mode: `{args.smoke}`",
        "",
        "| Dataset | Method | Seed | Dev F1 | Test P | Test R | Test F1 | O-bias | JSON |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        final = row["final_test"]
        best_dev = row["best_dev"]
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['seed']} | "
            f"{best_dev['f1']:.4f} | {final['precision']:.4f} | "
            f"{final['recall']:.4f} | {final['f1']:.4f} | "
            f"{row['best_decode_o_bias']:.3f} | `{Path(row['json']).name}` |"
        )
    lines.append("")
    LOG_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    specs = make_specs(args)
    paths = []
    for spec in specs:
        print(f"=== {spec['dataset']} seed={spec['seed']} {spec['label']} ===")
        paths.append(run_one(spec))
    rows = summarize(paths)
    write_tex(rows)
    write_log(rows, args)
    print(LOG_PATH)


if __name__ == "__main__":
    main()
