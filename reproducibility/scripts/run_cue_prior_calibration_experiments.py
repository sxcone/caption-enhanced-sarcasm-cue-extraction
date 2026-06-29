#!/usr/bin/env python3
"""Run cue-type prior calibrated decoding experiments.

This experiment extends O-bias calibration by tuning small per-cue label biases
on the dev set. It is designed for the project's cue-extraction datasets, where
each branch has a compact label inventory (text: TENT/TOPI; caption: MENT/MOPI).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path


PAPER_DIR = Path(__file__).resolve().parents[1]
ROOT = PAPER_DIR.parent
CODE_DIR = ROOT / "SRTP" / "HVFormer-main" / "HVFormer-main"
SRC_REPORT_DIR = CODE_DIR / "reports" / "baseline_results"
RAW_DIR = PAPER_DIR / "results" / "cue_prior_calibration" / "raw"
TABLE_DIR = PAPER_DIR / "tables"
RESULT_TABLE_DIR = PAPER_DIR / "results" / "tables"
LOG_PATH = PAPER_DIR / "results" / "cue_prior_calibration" / "experiment_log.md"
DEFAULT_PYTHON = Path("/Users/sxc/Desktop/去年的SRTP/paper_submission/.venv-mps/bin/python")
PYTHON = os.environ.get("PAPER_PYTHON") or (
    str(DEFAULT_PYTHON) if DEFAULT_PYTHON.exists() else (shutil.which("python3") or "python3")
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["caption_ner", "text_ner"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_seq", type=int, default=80)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--boundary_weight", type=float, default=0.4)
    parser.add_argument("--reuse_existing", action="store_true")
    return parser.parse_args()


def build_command(
    dataset: str,
    seed: int,
    experiment_name: str,
    args: argparse.Namespace,
    *,
    use_boundary: bool,
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
        str(args.num_epochs),
        "--batch_size",
        str(args.batch_size),
        "--max_seq",
        str(args.max_seq),
        "--device",
        args.device,
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
        "--tune_decode_o_bias",
        "--tune_label_biases",
        "--decode_o_bias_min",
        "-2.0",
        "--decode_o_bias_max",
        "2.0",
        "--decode_o_bias_step",
        "0.1",
        "--label_bias_min",
        "-0.5",
        "--label_bias_max",
        "0.5",
        "--label_bias_step",
        "0.25",
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
        "Cue-type prior calibrated decoding",
    ]
    if use_boundary:
        command.extend(["--use_span_boundary_head", "--boundary_loss_weight", str(args.boundary_weight)])
    return command


def latest_result(experiment_name: str, dataset: str) -> Path | None:
    candidates = list(RAW_DIR.glob(f"{experiment_name}_{dataset}_*.json"))
    candidates.extend(SRC_REPORT_DIR.glob(f"{experiment_name}_{dataset}_*.json"))
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: path.stat().st_mtime)[-1]


def run_one(spec: dict, reuse_existing: bool) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    source = latest_result(spec["experiment_name"], spec["dataset"])
    if source is not None:
        print(f"Reusing {source.name}")
    elif not reuse_existing:
        before = {p.resolve() for p in SRC_REPORT_DIR.glob(f"{spec['experiment_name']}_{spec['dataset']}_*.json")}
        subprocess.run(spec["command"], cwd=CODE_DIR, check=True)
        candidates = sorted(
            [
                p
                for p in SRC_REPORT_DIR.glob(f"{spec['experiment_name']}_{spec['dataset']}_*.json")
                if p.resolve() not in before
            ],
            key=lambda path: path.stat().st_mtime,
        )
        if not candidates:
            raise FileNotFoundError(f"No JSON result for {spec['experiment_name']} on {spec['dataset']}")
        source = candidates[-1]
    else:
        raise FileNotFoundError(f"No reusable JSON result for {spec['experiment_name']} on {spec['dataset']}")

    data = json.loads(source.read_text(encoding="utf-8"))
    data.update(
        {
            "paper_method_key": spec["key"],
            "paper_method_label": spec["label"],
            "paper_seed": spec["seed"],
            "paper_source_json": str(source),
        }
    )
    target = RAW_DIR / source.name
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def metric(data: dict, block: str, key: str) -> float:
    return float(data.get(block, {}).get(key, 0.0))


def summarize(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "dataset": data.get("dataset_name"),
                "method": data.get("paper_method_label"),
                "seed": data.get("paper_seed"),
                "dev_f1": metric(data, "best_dev_metrics", "f1"),
                "test_p": metric(data, "final_test_metrics", "precision"),
                "test_r": metric(data, "final_test_metrics", "recall"),
                "test_f1": metric(data, "final_test_metrics", "f1"),
                "o_bias": float(data.get("best_decode_o_bias", 0.0)),
                "label_biases": data.get("best_decode_label_biases", {}),
                "json": str(path),
            }
        )
    return sorted(rows, key=lambda row: (row["dataset"], row["seed"], row["method"]))


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return (sum((value - avg) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def aggregate(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["method"])].append(row)

    aggregate_rows = []
    for (dataset, method), items in grouped.items():
        aggregate_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "seeds": ",".join(str(item["seed"]) for item in sorted(items, key=lambda row: row["seed"])),
                "dev_f1": mean([item["dev_f1"] for item in items]),
                "test_p": mean([item["test_p"] for item in items]),
                "test_r": mean([item["test_r"] for item in items]),
                "test_f1": mean([item["test_f1"] for item in items]),
                "test_f1_std": std([item["test_f1"] for item in items]),
            }
        )
    return sorted(aggregate_rows, key=lambda row: (row["dataset"], row["method"]))


def latex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def write_outputs(rows: list[dict], args: argparse.Namespace) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    tex_lines = [
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "Dataset & Method & Seed & Dev F1 & Test P & Test R & Test F1 \\\\",
        "\\midrule",
    ]
    for row in rows:
        dataset = latex_escape(row["dataset"])
        tex_lines.append(
            f"{dataset} & {row['method']} & {row['seed']} & "
            f"{row['dev_f1']:.4f} & {row['test_p']:.4f} & {row['test_r']:.4f} & {row['test_f1']:.4f} \\\\"
        )
    tex_lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    tex = "\n".join(tex_lines)
    for out_dir in (TABLE_DIR, RESULT_TABLE_DIR):
        (out_dir / "cue_prior_calibration_results.tex").write_text(tex, encoding="utf-8")

    aggregate_rows = aggregate(rows)
    summary_lines = [
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Dataset & Method & Dev F1 & Test P & Test R & Test F1 \\\\",
        "\\midrule",
    ]
    for row in aggregate_rows:
        dataset = latex_escape(row["dataset"])
        summary_lines.append(
            f"{dataset} & {row['method']} & {row['dev_f1']:.4f} & "
            f"{row['test_p']:.4f} & {row['test_r']:.4f} & "
            f"{row['test_f1']:.4f}$\\pm${row['test_f1_std']:.4f} \\\\"
        )
    summary_lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    summary_tex = "\n".join(summary_lines)
    for out_dir in (TABLE_DIR, RESULT_TABLE_DIR):
        (out_dir / "cue_prior_calibration_summary.tex").write_text(summary_tex, encoding="utf-8")

    log_lines = [
        "# Cue-Type Prior Calibration Experiments",
        "",
        f"- Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"- Python: `{PYTHON}`",
        f"- Device: `{args.device}`",
        f"- Epochs: `{args.num_epochs}`",
        "",
        "| Dataset | Method | Seed | Dev F1 | Test P | Test R | Test F1 | O-bias | Label biases | JSON |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        log_lines.append(
            f"| {row['dataset']} | {row['method']} | {row['seed']} | {row['dev_f1']:.4f} | "
            f"{row['test_p']:.4f} | {row['test_r']:.4f} | {row['test_f1']:.4f} | "
            f"{row['o_bias']:.3f} | `{json.dumps(row['label_biases'], ensure_ascii=False)}` | "
            f"`{Path(row['json']).name}` |"
        )
    log_lines.extend(
        [
            "",
            "## Aggregate Results",
            "",
            "| Dataset | Method | Seeds | Dev F1 | Test P | Test R | Test F1 |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in aggregate_rows:
        log_lines.append(
            f"| {row['dataset']} | {row['method']} | {row['seeds']} | {row['dev_f1']:.4f} | "
            f"{row['test_p']:.4f} | {row['test_r']:.4f} | "
            f"{row['test_f1']:.4f} ± {row['test_f1_std']:.4f} |"
        )
    LOG_PATH.write_text("\n".join(log_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    specs = []
    for dataset in args.datasets:
        for seed in args.seeds:
            for key, label, use_boundary in [
                ("hvformer_cue_prior", "HVFormer + CuePrior", False),
                ("bcce_cue_prior", f"BC-CE $\\lambda={args.boundary_weight:g}$ + CuePrior", True),
            ]:
                experiment_name = f"cueprior_{key}_seed{seed}"
                specs.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "key": key,
                        "label": label,
                        "experiment_name": experiment_name,
                        "command": build_command(dataset, seed, experiment_name, args, use_boundary=use_boundary),
                    }
                )

    paths = []
    for spec in specs:
        print(f"=== {spec['dataset']} seed={spec['seed']} {spec['label']} ===")
        paths.append(run_one(spec, args.reuse_existing))
    rows = summarize(paths)
    write_outputs(rows, args)
    print(LOG_PATH)


if __name__ == "__main__":
    main()
