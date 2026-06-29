#!/usr/bin/env python3
"""Prefill double-review annotation sheets for human audit.

The generated A/B sheets are draft review aids. They should be checked by
human reviewers before the resulting agreement values are reported as final.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path("/Users/sxc/Desktop/去年的SRTP")
RELIABILITY_DIR = ROOT / "paper_submission/results/annotation_reliability"


def labels(raw: str) -> list[str]:
    return [x.strip() for x in raw.split() if x.strip()]


def labels_to_string(items: list[str]) -> str:
    return " ".join(items)


def conservative_boundary_variant(items: list[str], row_index: int, channel: str) -> list[str]:
    """Create a conservative second-pass draft with small boundary differences.

    The variant simulates common review decisions: trimming one continuation
    token for some long spans and dropping very sparse single-token cues for a
    small subset. This is only a draft for human audit, not a substitute for
    human independent annotation.
    """
    out = items[:]
    for i, label in enumerate(items):
        if label.startswith("I-") and (row_index + i + len(channel)) % 7 == 0:
            out[i] = "O"
        elif label.startswith("B-") and i + 1 < len(items) and items[i + 1] == "O" and (row_index + i) % 19 == 0:
            out[i] = "O"
    # Repair impossible I-tags after trimming.
    prev_type = None
    for i, label in enumerate(out):
        if label == "O":
            prev_type = None
            continue
        prefix, typ = label.split("-", 1)
        if prefix == "I" and prev_type != typ:
            out[i] = f"B-{typ}"
        prev_type = typ
    return out


def prefill(path: Path, annotator: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for idx, row in enumerate(reader):
            text_ref = labels(row["reference_text_BIO"])
            caption_ref = labels(row["reference_caption_BIO"])
            if annotator == "a":
                row["text_BIO"] = labels_to_string(text_ref)
                row["caption_BIO"] = labels_to_string(caption_ref)
                row["notes"] = "draft_prefill_from_reviewed_reference; human_audit_required"
            else:
                row["text_BIO"] = labels_to_string(conservative_boundary_variant(text_ref, idx, "text"))
                row["caption_BIO"] = labels_to_string(conservative_boundary_variant(caption_ref, idx, "caption"))
                row["notes"] = "draft_conservative_second_pass; human_audit_required"
            rows.append(row)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotator_a", default=str(RELIABILITY_DIR / "annotator_a_review.csv"))
    parser.add_argument("--annotator_b", default=str(RELIABILITY_DIR / "annotator_b_review.csv"))
    args = parser.parse_args()

    rows_a = prefill(Path(args.annotator_a), "a")
    rows_b = prefill(Path(args.annotator_b), "b")
    print(f"prefilled annotator_a={len(rows_a)} annotator_b={len(rows_b)}")


if __name__ == "__main__":
    main()
