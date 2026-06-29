#!/usr/bin/env python3
"""Evaluate VLM prompt cue lists by aligning phrases to BIO gold tokens."""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TEXT_FILE = ROOT / "caption和text" / "text.txt"
CAPTION_FILE = ROOT / "caption和text" / "caption.txt"
OUT_DIR = ROOT / "paper_submission" / "results" / "vlm_baseline"
SPLIT_SIZES = [("train", 1230), ("dev", 263), ("test", 265)]
TEXT_LABELS = {"TENT", "TOPI"}
CAPTION_LABELS = {"MENT", "MOPI"}
ALL_LABELS = ["TENT", "TOPI", "MENT", "MOPI"]


def parse_bio(path: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    imgid = None
    tokens: list[str] = []
    labels: list[str] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            if imgid is not None:
                samples.append({"imgid": imgid, "tokens": tokens, "labels": labels})
            imgid, tokens, labels = None, [], []
            continue
        if line.startswith("IMGID:"):
            if imgid is not None:
                samples.append({"imgid": imgid, "tokens": tokens, "labels": labels})
            imgid = line.split(":", 1)[1].strip()
            tokens, labels = [], []
            continue
        parts = line.rsplit(maxsplit=1)
        if len(parts) == 2:
            tokens.append(parts[0])
            labels.append(parts[1])
    if imgid is not None:
        samples.append({"imgid": imgid, "tokens": tokens, "labels": labels})
    return samples


def test_split(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cursor = 0
    for split, size in SPLIT_SIZES:
        part = samples[cursor : cursor + size]
        cursor += size
        if split == "test":
            return part
    raise RuntimeError("test split not found")


def normalize_token(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", t.lower())


def normalize_phrase(phrase: str) -> list[str]:
    return [x for x in (normalize_token(t) for t in phrase.split()) if x]


def spans_from_bio(labels: list[str], wanted: str) -> set[tuple[int, int]]:
    spans: set[tuple[int, int]] = set()
    start = None
    for i, label in enumerate(labels + ["O"]):
        if label == f"B-{wanted}":
            if start is not None:
                spans.add((start, i))
            start = i
        elif label == f"I-{wanted}" and start is not None:
            continue
        else:
            if start is not None:
                spans.add((start, i))
            start = None
    return spans


def token_gold(labels: list[str], wanted: str) -> set[int]:
    return {i for i, label in enumerate(labels) if label.endswith(f"-{wanted}")}


def align_phrase(tokens: list[str], phrase: str) -> list[tuple[int, int]]:
    phrase_tokens = normalize_phrase(phrase)
    if not phrase_tokens:
        return []
    norm_tokens = [normalize_token(t) for t in tokens]
    out: list[tuple[int, int]] = []
    n = len(phrase_tokens)
    for i in range(0, len(norm_tokens) - n + 1):
        if norm_tokens[i : i + n] == phrase_tokens:
            out.append((i, i + n))
    if out:
        return out
    # fallback: single-token substring matches for short VLM phrases
    joined_phrase = " ".join(phrase_tokens)
    for i, tok in enumerate(norm_tokens):
        if tok and (tok in phrase_tokens or tok in joined_phrase):
            out.append((i, i + 1))
    return out


def extract_json(raw: str) -> tuple[dict[str, Any], bool]:
    raw = raw.strip()
    try:
        obj = json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, flags=re.S)
        if not m:
            return {k: [] for k in ALL_LABELS} | {"rationale": ""}, False
        try:
            obj = json.loads(m.group(0))
        except Exception:
            try:
                obj = ast.literal_eval(m.group(0))
            except Exception:
                return {k: [] for k in ALL_LABELS} | {"rationale": ""}, False
    if isinstance(obj, list):
        merged: dict[str, Any] = {}
        for item in obj:
            if isinstance(item, dict):
                for k, v in item.items():
                    if str(k).upper() in ALL_LABELS:
                        merged.setdefault(str(k).upper(), [])
                        if isinstance(v, list):
                            merged[str(k).upper()].extend(v)
                        else:
                            merged[str(k).upper()].append(v)
        obj = merged
    if not isinstance(obj, dict):
        return {k: [] for k in ALL_LABELS} | {"rationale": ""}, False
    fixed: dict[str, Any] = {}
    lowered = {str(k).upper(): v for k, v in obj.items()}
    for key in ALL_LABELS:
        vals = lowered.get(key, [])
        if isinstance(vals, str):
            vals = [vals] if vals.strip() else []
        if not isinstance(vals, list):
            vals = []
        fixed[key] = [str(v).strip() for v in vals if str(v).strip()]
    fixed["rationale"] = str(obj.get("rationale", obj.get("RATIONALE", "")))
    return fixed, any(fixed[key] for key in ALL_LABELS)


def prf(tp: int, pred: int, gold: int) -> dict[str, float]:
    p = tp / pred if pred else 0.0
    r = tp / gold if gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default=str(OUT_DIR / "raw_outputs.jsonl"))
    parser.add_argument("--output_dir", default=str(OUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    text_samples = {s["imgid"]: s for s in test_split(parse_bio(TEXT_FILE))}
    caption_samples = {s["imgid"]: s for s in test_split(parse_bio(CAPTION_FILE))}

    rows = []
    for line in Path(args.predictions).read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            rows.append(json.loads(line))

    token_counts = Counter()
    exact_span_counts = Counter()
    partial_span_counts = Counter()
    per_class = defaultdict(Counter)
    parse_ok = 0
    alignment_failures = 0
    samples_with_pred = 0
    sample_coverage_hits = 0
    detailed_rows = []

    for row in rows:
        imgid = row["imgid"]
        pred_obj, ok = extract_json(row.get("raw_prediction", ""))
        parse_ok += int(ok)
        sample_pred_tokens: set[tuple[str, int]] = set()
        sample_gold_tokens: set[tuple[str, int]] = set()
        sample_aligned_any = False

        for label in ALL_LABELS:
            gold_sample = text_samples[imgid] if label in TEXT_LABELS else caption_samples[imgid]
            tokens = gold_sample["tokens"]
            labels = gold_sample["labels"]
            gold_tokens = token_gold(labels, label)
            gold_spans = spans_from_bio(labels, label)
            pred_spans: set[tuple[int, int]] = set()
            pred_tokens: set[int] = set()
            phrases = pred_obj.get(label, [])
            if phrases:
                samples_with_pred += 1
            for phrase in phrases:
                matched = align_phrase(tokens, phrase)
                if not matched:
                    alignment_failures += 1
                for start, end in matched:
                    sample_aligned_any = True
                    pred_spans.add((start, end))
                    pred_tokens.update(range(start, end))

            tp_tokens = len(pred_tokens & gold_tokens)
            token_counts["tp"] += tp_tokens
            token_counts["pred"] += len(pred_tokens)
            token_counts["gold"] += len(gold_tokens)
            per_class[label]["token_tp"] += tp_tokens
            per_class[label]["token_pred"] += len(pred_tokens)
            per_class[label]["token_gold"] += len(gold_tokens)

            exact_tp = len(pred_spans & gold_spans)
            exact_span_counts["tp"] += exact_tp
            exact_span_counts["pred"] += len(pred_spans)
            exact_span_counts["gold"] += len(gold_spans)
            per_class[label]["exact_tp"] += exact_tp
            per_class[label]["exact_pred"] += len(pred_spans)
            per_class[label]["exact_gold"] += len(gold_spans)

            partial_tp = 0
            used_gold: set[tuple[int, int]] = set()
            for ps in pred_spans:
                for gs in gold_spans:
                    if gs in used_gold:
                        continue
                    if max(ps[0], gs[0]) < min(ps[1], gs[1]):
                        partial_tp += 1
                        used_gold.add(gs)
                        break
            partial_span_counts["tp"] += partial_tp
            partial_span_counts["pred"] += len(pred_spans)
            partial_span_counts["gold"] += len(gold_spans)
            per_class[label]["partial_tp"] += partial_tp
            per_class[label]["partial_pred"] += len(pred_spans)
            per_class[label]["partial_gold"] += len(gold_spans)

            sample_pred_tokens.update((label, i) for i in pred_tokens)
            sample_gold_tokens.update((label, i) for i in gold_tokens)

        if sample_pred_tokens & sample_gold_tokens:
            sample_coverage_hits += 1
        detailed_rows.append(
            {
                "imgid": imgid,
                "parse_ok": ok,
                "aligned_any": sample_aligned_any,
                "raw_prediction": row.get("raw_prediction", ""),
                "parsed_prediction": pred_obj,
            }
        )

    class_summary = {}
    for label in ALL_LABELS:
        c = per_class[label]
        class_summary[label] = {
            "token": prf(c["token_tp"], c["token_pred"], c["token_gold"]),
            "exact_span": prf(c["exact_tp"], c["exact_pred"], c["exact_gold"]),
            "partial_span": prf(c["partial_tp"], c["partial_pred"], c["partial_gold"]),
        }

    phrase_count = sum(len(extract_json(r.get("raw_prediction", ""))[0].get(label, [])) for r in rows for label in ALL_LABELS)
    summary = {
        "samples_predicted": len(rows),
        "parse_success_rate": parse_ok / len(rows) if rows else 0.0,
        "phrase_count": phrase_count,
        "alignment_failure_rate": alignment_failures / phrase_count if phrase_count else 0.0,
        "sample_level_cue_coverage": sample_coverage_hits / len(rows) if rows else 0.0,
        "token_overlap": prf(token_counts["tp"], token_counts["pred"], token_counts["gold"]),
        "exact_span": prf(exact_span_counts["tp"], exact_span_counts["pred"], exact_span_counts["gold"]),
        "partial_span": prf(partial_span_counts["tp"], partial_span_counts["pred"], partial_span_counts["gold"]),
        "per_class": class_summary,
        "note": "VLM phrase outputs are aligned to gold token sequences; this diagnostic is not directly comparable to supervised sequence labellers.",
    }
    (output_dir / "vlm_baseline_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "vlm_baseline_detailed.json").write_text(
        json.dumps(detailed_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    with (output_dir / "vlm_baseline_table.tex").open("w", encoding="utf-8") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\caption{Local open-source VLM prompt baseline on the test split.}\n")
        f.write("\\label{tab:vlm-baseline}\n")
        f.write("\\scriptsize\n\\begin{tabular}{lrrr}\n\\toprule\n")
        f.write("Metric & Precision & Recall & F1 \\\\\n\\midrule\n")
        for name, key in [
            ("Token overlap", "token_overlap"),
            ("Exact span", "exact_span"),
            ("Partial span", "partial_span"),
        ]:
            vals = summary[key]
            f.write(
                f"{name} & {vals['precision'] * 100:.2f} & {vals['recall'] * 100:.2f} & {vals['f1'] * 100:.2f} \\\\\n"
            )
        f.write("\\midrule\n")
        f.write(
            "\\multicolumn{4}{l}{Parse success: "
            f"{summary['parse_success_rate'] * 100:.2f}\\%; alignment failure: {summary['alignment_failure_rate'] * 100:.2f}\\%; sample coverage: {summary['sample_level_cue_coverage'] * 100:.2f}\\%.}} \\\\\n"
        )
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    md = [
        "# Local VLM prompt baseline",
        "",
        f"- Predicted samples: {summary['samples_predicted']}",
        f"- Parse success rate: {summary['parse_success_rate'] * 100:.2f}%",
        f"- Alignment failure rate: {summary['alignment_failure_rate'] * 100:.2f}%",
        f"- Sample-level cue coverage: {summary['sample_level_cue_coverage'] * 100:.2f}%",
        f"- Token overlap F1: {summary['token_overlap']['f1'] * 100:.2f}",
        f"- Exact-span F1: {summary['exact_span']['f1'] * 100:.2f}",
        f"- Partial-span F1: {summary['partial_span']['f1'] * 100:.2f}",
        "",
        "This is a phrase-prompt diagnostic baseline. It should not be treated as a fully supervised sequence labeller.",
    ]
    (output_dir / "vlm_baseline_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
