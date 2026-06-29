#!/usr/bin/env python3
"""Run the 5-epoch core experiments used by the final paper tables."""

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
RAW_DIR = PAPER_DIR / "results" / "epoch5_core" / "raw"
TABLE_DIR = PAPER_DIR / "tables"
RESULT_TABLE_DIR = PAPER_DIR / "results" / "tables"
LOG_PATH = PAPER_DIR / "results" / "epoch5_core" / "experiment_log.md"
DEFAULT_PYTHON = PAPER_DIR / ".venv-mps" / "bin" / "python"
PYTHON = os.environ.get("PAPER_PYTHON") or (
    str(DEFAULT_PYTHON) if DEFAULT_PYTHON.exists() else (shutil.which("python3") or "python3")
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["caption_ner", "text_ner"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_seq", type=int, default=80)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--boundary_weight", type=float, default=0.4)
    parser.add_argument("--reuse_existing", action="store_true")
    return parser.parse_args()


def base_command(dataset: str, seed: int, experiment_name: str, args: argparse.Namespace) -> list[str]:
    return [
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
        "5-epoch core paper experiment",
    ]


def add_o_bias(command: list[str]) -> None:
    command.extend(
        [
            "--tune_decode_o_bias",
            "--decode_o_bias_min",
            "-2.0",
            "--decode_o_bias_max",
            "2.0",
            "--decode_o_bias_step",
            "0.1",
        ]
    )


def add_label_bias(command: list[str]) -> None:
    command.extend(
        [
            "--tune_label_biases",
            "--label_bias_min",
            "-0.5",
            "--label_bias_max",
            "0.5",
            "--label_bias_step",
            "0.25",
        ]
    )


def add_boundary(command: list[str], weight: float) -> None:
    command.extend(["--use_span_boundary_head", "--boundary_loss_weight", str(weight)])


def build_specs(args: argparse.Namespace) -> list[dict]:
    specs = []
    for dataset in args.datasets:
        for seed in args.seeds:
            variants = [
                ("hvformer", "HVFormer", False, False, False),
                ("hvformer_obias", "HVFormer + O-bias", False, True, False),
                ("bcce_obias", f"BC-CE $\\lambda={args.boundary_weight:g}$ + O-bias", True, True, False),
                ("hvformer_cueprior", "HVFormer + CuePrior", False, True, True),
                ("bcce_cueprior", f"BC-CE $\\lambda={args.boundary_weight:g}$ + CuePrior", True, True, True),
            ]
            for key, label, use_boundary, use_o_bias, use_label_bias in variants:
                experiment_name = f"epoch5_{key}_seed{seed}"
                command = base_command(dataset, seed, experiment_name, args)
                if use_boundary:
                    add_boundary(command, args.boundary_weight)
                if use_o_bias:
                    add_o_bias(command)
                if use_label_bias:
                    add_label_bias(command)
                specs.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "key": key,
                        "label": label,
                        "experiment_name": experiment_name,
                        "command": command,
                        "num_epochs": args.num_epochs,
                    }
                )
    return specs


def existing_result(spec: dict) -> Path | None:
    candidates = list(RAW_DIR.glob(f"{spec['experiment_name']}_{spec['dataset']}_*.json"))
    candidates.extend(SRC_REPORT_DIR.glob(f"{spec['experiment_name']}_{spec['dataset']}_*.json"))
    for path in sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if int(data.get("num_epochs", -1)) == int(spec["num_epochs"]):
            return path
    return None


def run_one(spec: dict, reuse_existing: bool) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    source = existing_result(spec)
    if source is not None:
        print(f"Reusing {source.name}")
    elif reuse_existing:
        raise FileNotFoundError(f"No reusable result for {spec['experiment_name']} on {spec['dataset']}")
    else:
        before = {p.resolve() for p in SRC_REPORT_DIR.glob(f"{spec['experiment_name']}_{spec['dataset']}_*.json")}
        subprocess.run(spec["command"], cwd=CODE_DIR, check=True)
        candidates = sorted(
            [
                p
                for p in SRC_REPORT_DIR.glob(f"{spec['experiment_name']}_{spec['dataset']}_*.json")
                if p.resolve() not in before
            ],
            key=lambda p: p.stat().st_mtime,
        )
        if not candidates:
            raise FileNotFoundError(f"No JSON result for {spec['experiment_name']} on {spec['dataset']}")
        source = candidates[-1]

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


def summarize(paths: list[Path]) -> tuple[list[dict], list[dict]]:
    rows = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "dataset": data["dataset_name"],
                "method": data["paper_method_label"],
                "method_key": data["paper_method_key"],
                "seed": int(data["paper_seed"]),
                "dev_f1": metric(data, "best_dev_metrics", "f1"),
                "test_p": metric(data, "final_test_metrics", "precision"),
                "test_r": metric(data, "final_test_metrics", "recall"),
                "test_f1": metric(data, "final_test_metrics", "f1"),
                "json": str(path),
            }
        )
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["method_key"], row["method"])].append(row)
    agg = []
    for (dataset, method_key, method), items in grouped.items():
        vals = [item["test_f1"] for item in items]
        avg = sum(vals) / len(vals)
        std = 0.0 if len(vals) < 2 else (sum((v - avg) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
        agg.append(
            {
                "dataset": dataset,
                "method_key": method_key,
                "method": method,
                "seeds": ",".join(str(item["seed"]) for item in sorted(items, key=lambda r: r["seed"])),
                "dev_f1": sum(item["dev_f1"] for item in items) / len(items),
                "test_p": sum(item["test_p"] for item in items) / len(items),
                "test_r": sum(item["test_r"] for item in items) / len(items),
                "test_f1": avg,
                "test_f1_std": std,
            }
        )
    order = {"hvformer": 0, "hvformer_obias": 1, "bcce_obias": 2, "hvformer_cueprior": 3, "bcce_cueprior": 4}
    rows.sort(key=lambda r: (r["dataset"], r["seed"], order.get(r["method_key"], 99)))
    agg.sort(key=lambda r: (r["dataset"], order.get(r["method_key"], 99)))
    return rows, agg


def latex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def write_outputs(rows: list[dict], agg: list[dict], args: argparse.Namespace) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    tex_lines = [
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Dataset & Method & Dev F1 & Test P & Test R & Test F1 \\\\",
        "\\midrule",
    ]
    for row in agg:
        tex_lines.append(
            f"{latex_escape(row['dataset'])} & {row['method']} & {row['dev_f1']:.4f} & "
            f"{row['test_p']:.4f} & {row['test_r']:.4f} & "
            f"{row['test_f1']:.4f}$\\pm${row['test_f1_std']:.4f} \\\\"
        )
    tex_lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    tex = "\n".join(tex_lines)
    for directory in (TABLE_DIR, RESULT_TABLE_DIR):
        (directory / "epoch5_core_results.tex").write_text(tex, encoding="utf-8")

    log_lines = [
        "# 5-Epoch Core Experiments",
        "",
        f"- Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"- Python: `{PYTHON}`",
        f"- Device: `{args.device}`",
        f"- Epochs: `{args.num_epochs}`",
        "",
        "## Aggregate",
        "",
        "| Dataset | Method | Seeds | Dev F1 | Test P | Test R | Test F1 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in agg:
        log_lines.append(
            f"| {row['dataset']} | {row['method']} | {row['seeds']} | {row['dev_f1']:.4f} | "
            f"{row['test_p']:.4f} | {row['test_r']:.4f} | "
            f"{row['test_f1']:.4f} ± {row['test_f1_std']:.4f} |"
        )
    log_lines.extend(
        [
            "",
            "## Per Seed",
            "",
            "| Dataset | Method | Seed | Dev F1 | Test P | Test R | Test F1 | JSON |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        log_lines.append(
            f"| {row['dataset']} | {row['method']} | {row['seed']} | {row['dev_f1']:.4f} | "
            f"{row['test_p']:.4f} | {row['test_r']:.4f} | {row['test_f1']:.4f} | "
            f"`{Path(row['json']).name}` |"
        )
    LOG_PATH.write_text("\n".join(log_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    paths = []
    for spec in build_specs(args):
        print(f"=== {spec['dataset']} seed={spec['seed']} {spec['label']} ===")
        paths.append(run_one(spec, args.reuse_existing))
    rows, agg = summarize(paths)
    write_outputs(rows, agg, args)
    print(LOG_PATH)


if __name__ == "__main__":
    main()
