import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parents[1]
REPORT_DIR = BASE_DIR / "reports" / "baseline_results"
FIGURE_DIR = BASE_DIR / "reports" / "figures"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="text_ner", choices=["text_ner", "caption_ner"])
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_seq", type=int, default=80)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--skip_baselines", action="store_true")
    return parser.parse_args()


def run_and_capture_json(command, expected_prefix):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    before = {p.resolve() for p in REPORT_DIR.glob("*.json")}
    subprocess.run(command, cwd=BASE_DIR, check=True)
    candidates = sorted(
        [p for p in REPORT_DIR.glob(f"{expected_prefix}*.json") if p.resolve() not in before],
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise RuntimeError(f"No new report JSON found for prefix: {expected_prefix}")
    return candidates[-1]


def build_main_command(args, experiment_name, extra_flags=None):
    extra_flags = extra_flags or []
    return [
        sys.executable,
        "run.py",
        "--do_train",
        "--model_name",
        "bert-vit-inter-ner",
        "--dataset_name",
        args.dataset_name,
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
        str(args.seed),
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
    ] + extra_flags


def build_baseline_command(args, model_type):
    return [
        sys.executable,
        "scripts/train_ner_baselines.py",
        "--dataset_name",
        args.dataset_name,
        "--model_type",
        model_type,
        "--num_epochs",
        str(args.num_epochs),
        "--batch_size",
        "16",
        "--max_seq",
        str(args.max_seq),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--log_steps",
        "20",
    ]


def read_f1(result_path):
    data = json.loads(result_path.read_text(encoding="utf-8"))
    return float(data["final_test_metrics"]["f1"]), data


def plot_barh(title, rows, out_path, color_main="#4f9ed8", color_rest="#a9d6e5"):
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "STHeiti", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    labels = [row["label"] for row in rows]
    scores = [row["f1"] for row in rows]
    colors = [color_main] + [color_rest] * (len(rows) - 1)

    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=180)
    fig.patch.set_facecolor("white")
    y = range(len(rows))
    bars = ax.barh(list(y), scores, color=colors, edgecolor="#2f78aa", alpha=0.95)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Test F1")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.28)
    max_score = max(scores) if scores else 1.0
    ax.set_xlim(0, max_score * 1.25 if max_score > 0 else 1.0)
    for bar, score in zip(bars, scores):
        ax.text(score + max(max_score * 0.02, 0.2), bar.get_y() + bar.get_height() / 2, f"{score:.4f}", va="center")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    text_only_spec = {
        "key": "hvformer_text_only",
        "label": "HVFormer Text-Only",
        "command": build_main_command(
            args,
            "suite_text_only",
            ["--num_epochs", str(args.num_epochs), "--batch_size", str(args.batch_size), "--lr", "3e-5", "--warmup_ratio", "0.06"],
        ),
        "prefix": f"suite_text_only_{args.dataset_name}_",
    }

    print(f"\n=== Running {text_only_spec['label']} ===")
    text_only_json = run_and_capture_json(text_only_spec["command"], text_only_spec["prefix"])
    text_only_f1, text_only_data = read_f1(text_only_json)
    text_only_ckpt = BASE_DIR / "ckpt" / text_only_data["experiment_name"]
    results = {
        text_only_spec["key"]: {
            "label": text_only_spec["label"],
            "f1": text_only_f1,
            "result_json": str(text_only_json),
            "best_dev_f1": float(text_only_data["best_dev_metrics"]["f1"]),
        }
    }
    print(f"{text_only_spec['label']}: test_f1={text_only_f1:.4f} ({text_only_json.name})")

    experiments = [
        {
            "key": "hvformer_multimodal",
            "label": "HVFormer Full",
            "command": build_main_command(
                args,
                "suite_full_mm",
                [
                    "--num_epochs",
                    "1",
                    "--batch_size",
                    str(args.batch_size),
                    "--lr",
                    "1e-5",
                    "--warmup_ratio",
                    "0.06",
                    "--load_path",
                    str(text_only_ckpt),
                    "--disable_fast_text_only",
                    "--disable_aux_images",
                    "--disable_rcnn_regions",
                ],
            ),
            "prefix": f"suite_full_mm_{args.dataset_name}_",
        },
        {
            "key": "hvformer_no_cross_modal",
            "label": "w/o Cross-Modal",
            "command": build_main_command(
                args,
                "suite_no_cross",
                [
                    "--num_epochs",
                    "1",
                    "--batch_size",
                    str(args.batch_size),
                    "--lr",
                    "1e-5",
                    "--warmup_ratio",
                    "0.06",
                    "--load_path",
                    str(text_only_ckpt),
                    "--disable_fast_text_only",
                    "--disable_aux_images",
                    "--disable_rcnn_regions",
                    "--disable_cross_modal",
                ],
            ),
            "prefix": f"suite_no_cross_{args.dataset_name}_",
        },
        {
            "key": "hvformer_no_moe",
            "label": "w/o MoE Fusion",
            "command": build_main_command(
                args,
                "suite_no_moe",
                [
                    "--num_epochs",
                    "1",
                    "--batch_size",
                    str(args.batch_size),
                    "--lr",
                    "1e-5",
                    "--warmup_ratio",
                    "0.06",
                    "--load_path",
                    str(text_only_ckpt),
                    "--disable_fast_text_only",
                    "--disable_aux_images",
                    "--disable_rcnn_regions",
                    "--disable_moe_fusion",
                ],
            ),
            "prefix": f"suite_no_moe_{args.dataset_name}_",
        },
    ]

    if not args.skip_baselines:
        experiments.extend(
            [
                {
                    "key": "bert_linear",
                    "label": "BERT-Linear",
                    "command": build_baseline_command(args, "bert_linear"),
                    "prefix": f"bert_linear_{args.dataset_name}_",
                },
                {
                    "key": "bert_crf",
                    "label": "BERT-CRF",
                    "command": build_baseline_command(args, "bert_crf"),
                    "prefix": f"bert_crf_{args.dataset_name}_",
                },
            ]
        )

    for spec in experiments:
        print(f"\n=== Running {spec['label']} ===")
        result_path = run_and_capture_json(spec["command"], spec["prefix"])
        f1, data = read_f1(result_path)
        results[spec["key"]] = {
            "label": spec["label"],
            "f1": f1,
            "result_json": str(result_path),
            "best_dev_f1": float(data["best_dev_metrics"]["f1"]),
        }
        print(f"{spec['label']}: test_f1={f1:.4f} ({result_path.name})")

    compare_rows = [
        results["hvformer_multimodal"],
        results["hvformer_text_only"],
    ]
    if "bert_linear" in results:
        compare_rows.append(results["bert_linear"])
    if "bert_crf" in results:
        compare_rows.append(results["bert_crf"])

    ablation_rows = [
        results["hvformer_multimodal"],
        results["hvformer_text_only"],
        results["hvformer_no_cross_modal"],
        results["hvformer_no_moe"],
    ]

    summary = {
        "dataset_name": args.dataset_name,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "device": args.device,
        "generated_at": stamp,
        "comparison": compare_rows,
        "ablation": ablation_rows,
    }

    summary_path = REPORT_DIR / f"{args.dataset_name}_multimodal_suite_{stamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    compare_figure = FIGURE_DIR / f"{args.dataset_name}_model_compare_{stamp}.png"
    ablation_figure = FIGURE_DIR / f"{args.dataset_name}_ablation_{stamp}.png"
    plot_barh(f"{args.dataset_name} Model Comparison", compare_rows, compare_figure)
    plot_barh(f"{args.dataset_name} Ablation Study", ablation_rows, ablation_figure)

    print("\n=== Suite complete ===")
    print(summary_path)
    print(compare_figure)
    print(ablation_figure)


if __name__ == "__main__":
    main()
