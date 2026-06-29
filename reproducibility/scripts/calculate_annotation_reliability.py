#!/usr/bin/env python3
"""Calculate double-review agreement metrics for sarcasm cue BIO labels."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def split_labels(raw: str) -> list[str]:
    return [x.strip() for x in raw.replace("|", " ").split() if x.strip()]


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return {row["review_id"]: row for row in csv.DictReader(f)}


def spans(labels: list[str]) -> set[tuple[int, int, str]]:
    out: set[tuple[int, int, str]] = set()
    start = None
    current_type = None
    for i, label in enumerate(labels + ["O"]):
        if label.startswith("B-"):
            if start is not None and current_type is not None:
                out.add((start, i, current_type))
            start = i
            current_type = label.split("-", 1)[1]
        elif label.startswith("I-") and current_type == label.split("-", 1)[1] and start is not None:
            continue
        else:
            if start is not None and current_type is not None:
                out.add((start, i, current_type))
            start = None
            current_type = None
    return out


def prf(tp: int, pred: int, gold: int) -> dict[str, float]:
    precision = tp / pred if pred else 0.0
    recall = tp / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def cohen_kappa(a: list[str], b: list[str]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    labels = sorted(set(a) | set(b))
    observed = sum(x == y for x, y in zip(a, b)) / len(a)
    ca = Counter(a)
    cb = Counter(b)
    expected = sum((ca[label] / len(a)) * (cb[label] / len(b)) for label in labels)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1 - expected)


def validate_and_collect(rows_a: dict[str, dict[str, str]], rows_b: dict[str, dict[str, str]]) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    collected: list[dict[str, Any]] = []
    for review_id, row_a in rows_a.items():
        row_b = rows_b.get(review_id)
        if row_b is None:
            errors.append(f"{review_id}: missing in annotator B")
            continue
        for channel in ["text", "caption"]:
            key = f"{channel}_BIO"
            labels_a = split_labels(row_a.get(key, ""))
            labels_b = split_labels(row_b.get(key, ""))
            ref = split_labels(row_a.get(f"reference_{channel}_BIO", ""))
            if not labels_a or not labels_b:
                errors.append(f"{review_id}: missing {channel} labels")
                continue
            if len(labels_a) != len(labels_b):
                errors.append(f"{review_id}: {channel} label length mismatch A={len(labels_a)} B={len(labels_b)}")
                continue
            if ref and len(labels_a) != len(ref):
                errors.append(f"{review_id}: {channel} label length differs from reference len={len(ref)}")
                continue
            collected.append({"review_id": review_id, "channel": channel, "a": labels_a, "b": labels_b})
    return collected, errors


def calculate(collected: list[dict[str, Any]]) -> dict[str, Any]:
    flat_a = [label for row in collected for label in row["a"]]
    flat_b = [label for row in collected for label in row["b"]]
    exact_tp = exact_a = exact_b = 0
    boundary_tp = boundary_a = boundary_b = 0
    pos_tp = pos_a = pos_b = 0
    per_class: dict[str, Counter] = {}

    for row in collected:
        a_labels = row["a"]
        b_labels = row["b"]
        a_spans = spans(a_labels)
        b_spans = spans(b_labels)
        exact_tp += len(a_spans & b_spans)
        exact_a += len(a_spans)
        exact_b += len(b_spans)
        a_boundaries = {(s, e) for s, e, _ in a_spans}
        b_boundaries = {(s, e) for s, e, _ in b_spans}
        boundary_tp += len(a_boundaries & b_boundaries)
        boundary_a += len(a_boundaries)
        boundary_b += len(b_boundaries)

        a_pos = {i for i, label in enumerate(a_labels) if label != "O"}
        b_pos = {i for i, label in enumerate(b_labels) if label != "O"}
        pos_tp += len(a_pos & b_pos)
        pos_a += len(a_pos)
        pos_b += len(b_pos)

        for label in sorted(set(a_labels) | set(b_labels)):
            if label == "O":
                continue
            per_class.setdefault(label, Counter())
            a_set = {i for i, x in enumerate(a_labels) if x == label}
            b_set = {i for i, x in enumerate(b_labels) if x == label}
            per_class[label]["tp"] += len(a_set & b_set)
            per_class[label]["a"] += len(a_set)
            per_class[label]["b"] += len(b_set)

    return {
        "review_units": len(collected),
        "tokens": len(flat_a),
        "token_cohen_kappa": cohen_kappa(flat_a, flat_b),
        "token_agreement": sum(x == y for x, y in zip(flat_a, flat_b)) / max(1, len(flat_a)),
        "positive_token_agreement": prf(pos_tp, pos_a, pos_b),
        "exact_span_agreement": prf(exact_tp, exact_a, exact_b),
        "boundary_agreement": prf(boundary_tp, boundary_a, boundary_b),
        "per_class_token_agreement": {
            label: prf(counts["tp"], counts["a"], counts["b"])
            for label, counts in sorted(per_class.items())
        },
    }


def write_outputs(results: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "annotation_reliability_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    md = [
        "# Annotation reliability results",
        "",
        f"- Review units: {results['review_units']}",
        f"- Tokens: {results['tokens']}",
        f"- Token-level Cohen's kappa: {results['token_cohen_kappa']:.4f}",
        f"- Token agreement: {results['token_agreement'] * 100:.2f}%",
        f"- Positive-token F1: {results['positive_token_agreement']['f1'] * 100:.2f}",
        f"- Exact-span F1: {results['exact_span_agreement']['f1'] * 100:.2f}",
        f"- Boundary F1: {results['boundary_agreement']['f1'] * 100:.2f}",
    ]
    (output_dir / "annotation_reliability_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    tex = (
        "\\begin{tabular}{lr}\n"
        "\\toprule\n"
        "Metric & Value \\\\\n"
        "\\midrule\n"
        f"Token-level Cohen's $\\kappa$ & {results['token_cohen_kappa']:.4f} \\\\\n"
        f"Token agreement & {results['token_agreement'] * 100:.2f}\\% \\\\\n"
        f"Positive-token F1 & {results['positive_token_agreement']['f1'] * 100:.2f} \\\\\n"
        f"Exact-span F1 & {results['exact_span_agreement']['f1'] * 100:.2f} \\\\\n"
        f"Boundary F1 & {results['boundary_agreement']['f1'] * 100:.2f} \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )
    (output_dir / "annotation_reliability_table.tex").write_text(tex, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotator_a", required=True)
    parser.add_argument("--annotator_b", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--allow_incomplete", action="store_true")
    args = parser.parse_args()

    rows_a = read_rows(Path(args.annotator_a))
    rows_b = read_rows(Path(args.annotator_b))
    collected, errors = validate_and_collect(rows_a, rows_b)
    if errors and not args.allow_incomplete:
        print(json.dumps({
            "status": "waiting_for_complete_double_review",
            "usable_review_units": len(collected),
            "error_count": len(errors),
            "first_errors": errors[:20],
        }, indent=2, ensure_ascii=False))
        return
    results = calculate(collected)
    if errors:
        results["warnings"] = errors
    write_outputs(results, Path(args.output_dir))
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
