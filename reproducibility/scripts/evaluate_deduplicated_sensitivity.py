#!/usr/bin/env python3
"""Evaluate cue-extraction predictions under deduplicated test masks."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "paper_submission/results/deduplicated_sensitivity"
DEFAULT_TEXT_PAIRS = ROOT / "paper_submission/results/data_extension/near_duplicate_text_pairs.csv"
DEFAULT_IMAGE_PAIRS = ROOT / "paper_submission/results/data_extension/near_duplicate_image_pairs.csv"
DEFAULT_BIO_ROOT = ROOT / "caption和text"

TEXT_SPLITS = {
    "train": "text_part1_1000_580.txt",
    "dev": "text_part2_650_364.txt",
    "test": "text_3545_part5_595_394.txt",
}


def read_bio_samples(path: Path) -> list[dict]:
    samples = []
    cur_imgid = None
    cur_tokens = []
    cur_labels = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            if cur_imgid is not None and cur_tokens:
                samples.append({"imgid": cur_imgid, "tokens": cur_tokens, "labels": cur_labels})
            cur_imgid = None
            cur_tokens = []
            cur_labels = []
            continue
        if line.startswith("IMGID:"):
            if cur_imgid is not None and cur_tokens:
                samples.append({"imgid": cur_imgid, "tokens": cur_tokens, "labels": cur_labels})
            cur_imgid = line.split(":", 1)[1].strip()
            cur_tokens = []
            cur_labels = []
            continue
        parts = line.split("\t")
        if len(parts) == 2:
            cur_tokens.append(parts[0].lower())
            cur_labels.append(parts[1])
    if cur_imgid is not None and cur_tokens:
        samples.append({"imgid": cur_imgid, "tokens": cur_tokens, "labels": cur_labels})
    return samples


def load_current_text_splits(bio_root: Path) -> dict[str, list[dict]]:
    return {split: read_bio_samples(bio_root / name) for split, name in TEXT_SPLITS.items()}


def token_jaccard(a: list[str], b: list[str]) -> float:
    set_a = set(a)
    set_b = set(b)
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b) if set_a or set_b else 0.0


def discover_images() -> dict[str, Path]:
    image_index: dict[str, Path] = {}
    for pattern in ("part*_img_*", "SRTP/part*_img_*", "github_release/caption-enhanced-sarcasm-cue-extraction/part*_img_*"):
        for candidate in ROOT.glob(pattern):
            if not candidate.is_dir():
                continue
            nested = candidate / candidate.name
            target = nested if nested.is_dir() else candidate
            for image_path in target.iterdir():
                if image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                    image_index.setdefault(image_path.stem, image_path)
    return image_index


def ahash(path: Path, size: int = 8) -> int | None:
    try:
        with Image.open(path) as image:
            gray = image.convert("L").resize((size, size))
            values = list(gray.getdata())
    except Exception:
        return None
    avg = sum(values) / len(values)
    bits = 0
    for idx, value in enumerate(values):
        if value >= avg:
            bits |= 1 << idx
    return bits


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def build_current_split_masks(bio_root: Path, output_dir: Path) -> tuple[set[str], set[str], dict]:
    splits = load_current_text_splits(bio_root)
    train_dev = splits["train"] + splits["dev"]
    text_pairs = []
    text_ids = set()
    for test_sample in splits["test"]:
        for source_sample in train_dev:
            score = token_jaccard(test_sample["tokens"], source_sample["tokens"])
            if score >= 1.0:
                text_ids.add(test_sample["imgid"])
                text_pairs.append(
                    {
                        "sample_a": f"test:{test_sample['imgid']}",
                        "sample_b": f"train/dev:{source_sample['imgid']}",
                        "jaccard": f"{score:.4f}",
                    }
                )

    image_index = discover_images()
    ids_by_split = {split: [sample["imgid"] for sample in rows] for split, rows in splits.items()}
    needed_ids = set(ids_by_split["train"]) | set(ids_by_split["dev"]) | set(ids_by_split["test"])
    hashes = {imgid: ahash(image_index[imgid]) for imgid in needed_ids if imgid in image_index}
    hashes = {imgid: value for imgid, value in hashes.items() if value is not None}
    source_ids = ids_by_split["train"] + ids_by_split["dev"]
    image_pairs = []
    image_ids = set()
    for test_id in ids_by_split["test"]:
        if test_id not in hashes:
            continue
        for source_id in source_ids:
            if source_id not in hashes:
                continue
            distance = hamming(hashes[test_id], hashes[source_id])
            if distance <= 4:
                image_ids.add(test_id)
                image_pairs.append(
                    {
                        "sample_a": f"test:{test_id}",
                        "sample_b": f"train/dev:{source_id}",
                        "ahash_hamming": distance,
                    }
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "current_split_near_duplicate_text_pairs.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_a", "sample_b", "jaccard"])
        writer.writeheader()
        writer.writerows(text_pairs)
    with (output_dir / "current_split_near_duplicate_image_pairs.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_a", "sample_b", "ahash_hamming"])
        writer.writeheader()
        writer.writerows(image_pairs)
    details = {
        "bio_root": str(bio_root),
        "split_sizes": {split: len(rows) for split, rows in splits.items()},
        "text_leakage_pairs": len(text_pairs),
        "text_leakage_test_ids": sorted(text_ids),
        "perceptual_overlap_pairs": len(image_pairs),
        "perceptual_overlap_test_ids": sorted(image_ids),
        "image_hashes_available": len(hashes),
    }
    return text_ids, image_ids, details


def extract_split_id(value: str) -> tuple[str, str] | None:
    match = re.match(r"([^:]+):(.+)$", value.strip())
    if not match:
        return None
    return match.group(1), match.group(2)


def read_test_ids_from_pairs(path: Path, kind: str) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in ("sample_a", "sample_b"):
                parsed = extract_split_id(row.get(key, ""))
                if parsed and parsed[0] == "test":
                    ids.add(parsed[1])
    return ids


def spans(seq: list[str]) -> set[tuple[str, int, int]]:
    out = set()
    start = None
    label = None
    for idx, tag in enumerate(seq + ["O"]):
        if tag == "O":
            if label is not None:
                out.add((label, start, idx - 1))
                start = None
                label = None
            continue
        if label is None:
            start = idx
            label = tag
        elif tag != label:
            out.add((label, start, idx - 1))
            start = idx
            label = tag
    return out


def prf(tp: int, pred_total: int, gold_total: int) -> dict[str, float]:
    precision = tp / pred_total if pred_total else 0.0
    recall = tp / gold_total if gold_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": 100 * precision, "recall": 100 * recall, "f1": 100 * f1}


def compute_metrics(rows: list[dict]) -> dict:
    token_tp = token_pred = token_gold = 0
    span_tp = span_pred = span_gold = 0
    for sent_id, row in enumerate(rows):
        gold = row["gold_labels"]
        pred = row["pred_labels"]
        for g, p in zip(gold, pred):
            if p != "O":
                token_pred += 1
            if g != "O":
                token_gold += 1
            if g != "O" and g == p:
                token_tp += 1
        gold_spans = {(sent_id, *s) for s in spans(gold)}
        pred_spans = {(sent_id, *s) for s in spans(pred)}
        span_tp += len(gold_spans & pred_spans)
        span_pred += len(pred_spans)
        span_gold += len(gold_spans)
    token = prf(token_tp, token_pred, token_gold)
    span = prf(span_tp, span_pred, span_gold)
    return {
        "n": len(rows),
        "token_precision": token["precision"],
        "token_recall": token["recall"],
        "token_f1": token["f1"],
        "span_precision": span["precision"],
        "span_recall": span["recall"],
        "span_f1": span["f1"],
    }


def load_predictions(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def setting_rows(rows: list[dict], remove_ids: set[str]) -> list[dict]:
    return [row for row in rows if str(row["imgid"]) not in remove_ids]


def summarise(paths: list[Path], masks: dict[str, set[str]]) -> dict:
    per_run = []
    for path in paths:
        rows = load_predictions(path)
        if not rows:
            continue
        first = rows[0]
        model_key = "RoBERTa+Ctx" if first.get("bert_name") == "roberta-base" else "BERT-Linear"
        if first.get("paired_context") not in (None, "none"):
            model_key = "RoBERTa+Ctx"
        for setting, remove_ids in masks.items():
            kept = setting_rows(rows, remove_ids)
            metrics = compute_metrics(kept)
            metrics.update(
                {
                    "model": model_key,
                    "setting": setting,
                    "seed": first.get("seed"),
                    "prediction_file": str(path),
                    "removed_n": len(rows) - len(kept),
                    "original_n": len(rows),
                }
            )
            per_run.append(metrics)
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in per_run:
        grouped[(row["model"], row["setting"])].append(row)
    aggregate = []
    for (model, setting), rows in sorted(grouped.items()):
        original = next((r for r in grouped.get((model, "Original test"), []) if r["seed"] == rows[0]["seed"]), None)
        token_values = [r["token_f1"] for r in rows]
        span_values = [r["span_f1"] for r in rows]
        original_token_values = [r["token_f1"] for r in grouped.get((model, "Original test"), [])]
        original_span_values = [r["span_f1"] for r in grouped.get((model, "Original test"), [])]
        aggregate.append(
            {
                "model": model,
                "setting": setting,
                "seeds": sorted(r["seed"] for r in rows),
                "n": int(round(mean(r["n"] for r in rows))),
                "removed_n": int(round(mean(r["removed_n"] for r in rows))),
                "token_f1_mean": mean(token_values),
                "token_f1_std": pstdev(token_values) if len(token_values) > 1 else 0.0,
                "span_f1_mean": mean(span_values),
                "span_f1_std": pstdev(span_values) if len(span_values) > 1 else 0.0,
                "delta_token": mean(token_values) - mean(original_token_values) if original_token_values else 0.0,
                "delta_span": mean(span_values) - mean(original_span_values) if original_span_values else 0.0,
            }
        )
    return {"per_run": per_run, "aggregate": aggregate}


def write_table(summary: dict, path: Path) -> None:
    order = ["Original test", "Remove text leakage", "Remove perceptual overlap", "Deduplicated test"]
    rows = sorted(summary["aggregate"], key=lambda r: (r["model"] != "RoBERTa+Ctx", order.index(r["setting"])))
    setting_alias = {
        "Original test": "Original",
        "Remove text leakage": "No text leak",
        "Remove perceptual overlap": "No image overlap",
        "Deduplicated test": "Deduplicated",
    }
    lines = [
        "\\begin{table}[htbp]",
        "\\caption{Deduplicated-test sensitivity analysis. Delta values are relative to the original test setting.}",
        "\\label{tab:dedup-sensitivity}",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep\\fill}llrrrrrr}",
        "\\toprule",
        "Model & Setting & Test n & Removed & Token F1 & Span F1 & $\\Delta$ Token & $\\Delta$ Span \\\\",
        "\\midrule",
    ]
    for row in rows:
        token = f'{row["token_f1_mean"]:.2f}$\\pm${row["token_f1_std"]:.2f}'
        span = f'{row["span_f1_mean"]:.2f}$\\pm${row["span_f1_std"]:.2f}'
        lines.append(
            f'{row["model"]} & {setting_alias[row["setting"]]} & {row["n"]} & {row["removed_n"]} & {token} & {span} & {row["delta_token"]:+.2f} & {row["delta_span"]:+.2f} \\\\'
        )
    lines += ["\\botrule", "\\end{tabular*}", "\\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_dir", default=str(DEFAULT_RESULTS / "raw"))
    parser.add_argument("--output_dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--text_pairs", default=str(DEFAULT_TEXT_PAIRS))
    parser.add_argument("--image_pairs", default=str(DEFAULT_IMAGE_PAIRS))
    parser.add_argument("--bio_root", default=str(DEFAULT_BIO_ROOT))
    parser.add_argument("--use_legacy_pair_csv", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.use_legacy_pair_csv:
        text_ids = read_test_ids_from_pairs(Path(args.text_pairs), "text")
        image_ids = read_test_ids_from_pairs(Path(args.image_pairs), "image")
        mask_details = {
            "source": "legacy_pair_csv",
            "text_pairs": str(args.text_pairs),
            "image_pairs": str(args.image_pairs),
            "text_leakage_test_ids": sorted(text_ids),
            "perceptual_overlap_test_ids": sorted(image_ids),
        }
    else:
        text_ids, image_ids, mask_details = build_current_split_masks(Path(args.bio_root), output_dir)
    masks = {
        "Original test": set(),
        "Remove text leakage": text_ids,
        "Remove perceptual overlap": image_ids,
        "Deduplicated test": text_ids | image_ids,
    }
    (output_dir / "dedup_masks.json").write_text(
        json.dumps({"masks": {k: sorted(v) for k, v in masks.items()}, "details": mask_details}, indent=2),
        encoding="utf-8",
    )
    paths = sorted(Path(args.predictions_dir).glob("*_test_predictions.jsonl"))
    summary = summarise(paths, masks)
    (output_dir / "dedup_sensitivity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_table(summary, output_dir / "dedup_sensitivity_table.tex")
    md_lines = ["# Deduplicated-test sensitivity", "", "## Mask construction", ""]
    md_lines.append(json.dumps(mask_details, indent=2))
    md_lines.extend(["", "## Results", ""])
    for row in summary["aggregate"]:
        md_lines.append(
            f"- {row['model']} / {row['setting']}: n={row['n']}, Token F1={row['token_f1_mean']:.2f}, "
            f"Span F1={row['span_f1_mean']:.2f}, Delta span={row['delta_span']:+.2f}"
        )
    (output_dir / "experiment_log.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(output_dir / "dedup_sensitivity_summary.json")


if __name__ == "__main__":
    main()
