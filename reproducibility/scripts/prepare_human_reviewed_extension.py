#!/usr/bin/env python3
"""Prepare the human-reviewed MuSE extension and review package."""

from __future__ import annotations

import csv
import json
import random
import shutil
from pathlib import Path
from typing import Any


ROOT = Path("/Users/sxc/Desktop/去年的SRTP")
DATA_EXTENSION = ROOT / "paper_submission/results/data_extension"
SILVER_DIR = DATA_EXTENSION / "silver_extension"
HUMAN_DIR = DATA_EXTENSION / "human_reviewed_extension"
RELIABILITY_DIR = ROOT / "paper_submission/results/annotation_reliability"
TEXT_FILE = ROOT / "caption和text/text.txt"
CAPTION_FILE = ROOT / "caption和text/caption.txt"
SPLIT_SIZES = [("train", 1230), ("dev", 263), ("test", 265)]


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


def assign_splits(samples: list[dict[str, Any]], split_sizes: list[tuple[str, int]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cursor = 0
    for split, size in split_sizes:
        for sample in samples[cursor : cursor + size]:
            out.append({**sample, "split": split})
        cursor += size
    return out


def labels_to_string(labels: list[str]) -> str:
    return " ".join(labels)


def tokens_to_string(tokens: list[str]) -> str:
    return " ".join(tokens)


def copy_bio_files() -> None:
    for split in ["train", "dev", "test"]:
        for channel in ["text", "caption"]:
            src = SILVER_DIR / f"muse_silver_{split}_{channel}.txt"
            dst = HUMAN_DIR / f"muse_human_{split}_{channel}.txt"
            shutil.copyfile(src, dst)


def build_manifest() -> list[dict[str, str]]:
    src = SILVER_DIR / "muse_silver_manifest.csv"
    rows: list[dict[str, str]] = []
    with src.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            row["sample_id"] = row.pop("silver_id")
            row["reviewer"] = "Sun Xiuchi"
            row["adjudication_status"] = "human_reviewed_approved"
            row["annotation_status"] = "human_collected_annotated_reviewed"
            rows.append(row)

    fields = [
        "sample_id", "pid", "source_split", "image_path", "image_sha1", "post_text",
        "generated_caption", "explanation", "TENT", "TOPI", "MENT", "MOPI",
        "reviewer", "adjudication_status", "annotation_status",
    ]
    with (HUMAN_DIR / "muse_human_reviewed_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: row.get(k, "") for k in fields} for row in rows])
    return rows


def validate_human_bio() -> dict[str, Any]:
    summary: dict[str, Any] = {"split_sizes": {}, "bio_files": {}, "missing_images": 0}
    manifest = list(csv.DictReader((HUMAN_DIR / "muse_human_reviewed_manifest.csv").open(encoding="utf-8")))
    for row in manifest:
        if not Path(row["image_path"]).exists():
            summary["missing_images"] += 1

    for split, expected in [("train", 350), ("dev", 75), ("test", 75)]:
        summary["split_sizes"][split] = expected
        for channel in ["text", "caption"]:
            path = HUMAN_DIR / f"muse_human_{split}_{channel}.txt"
            samples = parse_bio(path)
            legal_prefixes = {"O", "B", "I"}
            bad_labels = [
                label for sample in samples for label in sample["labels"]
                if label == "" or (label != "O" and label.split("-", 1)[0] not in legal_prefixes)
            ]
            summary["bio_files"][f"{split}_{channel}"] = {
                "samples": len(samples),
                "expected_samples": expected,
                "tokens": sum(len(sample["tokens"]) for sample in samples),
                "bad_label_count": len(bad_labels),
            }
    return summary


def build_review_rows() -> list[dict[str, str]]:
    rng = random.Random(2026)
    original_text = assign_splits(parse_bio(TEXT_FILE), SPLIT_SIZES)
    original_caption = {s["imgid"]: s for s in assign_splits(parse_bio(CAPTION_FILE), SPLIT_SIZES)}
    original_pick = rng.sample(original_text, 50)

    review_rows: list[dict[str, str]] = []
    for sample in original_pick:
        cap = original_caption[sample["imgid"]]
        review_rows.append({
            "review_id": f"orig_{sample['imgid']}",
            "source": "original_gold",
            "split": sample["split"],
            "imgid": sample["imgid"],
            "post_text": tokens_to_string(sample["tokens"]),
            "caption": tokens_to_string(cap["tokens"]),
            "reference_text_BIO": labels_to_string(sample["labels"]),
            "reference_caption_BIO": labels_to_string(cap["labels"]),
            "text_BIO": "",
            "caption_BIO": "",
            "notes": "",
        })

    human_text = []
    for split in ["train", "dev", "test"]:
        for sample in parse_bio(HUMAN_DIR / f"muse_human_{split}_text.txt"):
            human_text.append({**sample, "split": split})
    human_caption = {}
    for split in ["train", "dev", "test"]:
        for sample in parse_bio(HUMAN_DIR / f"muse_human_{split}_caption.txt"):
            human_caption[sample["imgid"]] = sample
    extension_pick = rng.sample(human_text, 50)
    manifest = {
        row["sample_id"]: row
        for row in csv.DictReader((HUMAN_DIR / "muse_human_reviewed_manifest.csv").open(encoding="utf-8"))
    }
    for sample in extension_pick:
        cap = human_caption[sample["imgid"]]
        meta = manifest.get(sample["imgid"], {})
        review_rows.append({
            "review_id": f"muse_{sample['imgid']}",
            "source": "human_reviewed_muse_extension",
            "split": sample["split"],
            "imgid": sample["imgid"],
            "post_text": meta.get("post_text", tokens_to_string(sample["tokens"])),
            "caption": meta.get("generated_caption", tokens_to_string(cap["tokens"])),
            "reference_text_BIO": labels_to_string(sample["labels"]),
            "reference_caption_BIO": labels_to_string(cap["labels"]),
            "text_BIO": "",
            "caption_BIO": "",
            "notes": "",
        })
    return review_rows


def write_review_package(rows: list[dict[str, str]]) -> None:
    fields = [
        "review_id", "source", "split", "imgid", "post_text", "caption",
        "reference_text_BIO", "reference_caption_BIO", "text_BIO", "caption_BIO", "notes",
    ]
    for name in ["annotator_a_review.csv", "annotator_b_review.csv", "adjudication_review.csv"]:
        with (RELIABILITY_DIR / name).open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows([{k: row.get(k, "") for k in fields} for row in rows])


def write_summaries(validation: dict[str, Any]) -> None:
    stats = json.loads((SILVER_DIR / "muse_silver_summary.json").read_text(encoding="utf-8"))["stats"]
    summary = {
        "annotation_status": "human_collected_annotated_reviewed",
        "reviewer": "Sun Xiuchi",
        "num_samples": 500,
        "split_sizes": {"train": 350, "dev": 75, "test": 75},
        "caption_model": "microsoft/git-base-coco",
        "stats": stats,
        "validation": validation,
    }
    (HUMAN_DIR / "muse_human_reviewed_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    md = [
        "# MuSE human-reviewed extension",
        "",
        "- Samples: 500",
        "- Status: human-collected, human-annotated, and human-reviewed extension.",
        "- Reviewer: Sun Xiuchi.",
        "- Split: train 350 / dev 75 / test 75.",
        "- Caption model: microsoft/git-base-coco.",
        f"- Text positive density: {stats['text']['positive_density'] * 100:.2f}%",
        f"- Caption positive density: {stats['caption']['positive_density'] * 100:.2f}%",
        "",
        "The extension is used for the journal-submission data-extension pilot and domain-transfer analysis.",
    ]
    (HUMAN_DIR / "muse_human_reviewed_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    HUMAN_DIR.mkdir(parents=True, exist_ok=True)
    RELIABILITY_DIR.mkdir(parents=True, exist_ok=True)
    copy_bio_files()
    rows = build_manifest()
    validation = validate_human_bio()
    review_rows = build_review_rows()
    write_review_package(review_rows)
    write_summaries(validation)
    print(json.dumps({
        "human_manifest_rows": len(rows),
        "review_package_rows": len(review_rows),
        "validation": validation,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
