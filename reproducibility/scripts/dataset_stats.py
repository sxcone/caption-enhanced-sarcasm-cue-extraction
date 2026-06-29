#!/usr/bin/env python3
"""Build dataset statistics used by the paper."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "caption和text"
OUT_DIR = PAPER_DIR / "results"

SPLITS = {
    "train": {
        "text": DATA_DIR / "text_part1_1000_580.txt",
        "caption": DATA_DIR / "caption_3545_part1_1000_580.txt",
    },
    "dev": {
        "text": DATA_DIR / "text_part2_650_364.txt",
        "caption": DATA_DIR / "caption_part2_650_364.txt",
    },
    "test": {
        "text": DATA_DIR / "text_3545_part5_595_394.txt",
        "caption": DATA_DIR / "caption_3545_part5_595_394.txt",
    },
}

CAPTION_LABEL_ALIASES = {
    "OS": "O",
    "N-MOPI": "I-MOPI",
    "B-TENT": "B-MENT",
    "I-TENT": "I-MENT",
    "B-TOPI": "B-MOPI",
    "I-TOPI": "I-MOPI",
}


def parse_bio_file(path: Path, domain: str) -> dict:
    samples = 0
    tokens = 0
    image_ids: set[str] = set()
    labels: Counter[str] = Counter()
    bad_lines = 0

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("IMGID:"):
            samples += 1
            image_ids.add(line.split(":", 1)[1].strip())
            continue
        parts = line.rsplit(None, 1)
        if len(parts) != 2:
            bad_lines += 1
            continue
        tokens += 1
        label = parts[1]
        if domain == "caption":
            label = CAPTION_LABEL_ALIASES.get(label, label)
        labels[label] += 1

    return {
        "path": str(path),
        "samples": samples,
        "tokens": tokens,
        "unique_images": len(image_ids),
        "labels": dict(sorted(labels.items())),
        "bad_lines": bad_lines,
    }


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def write_dataset_table(stats: dict) -> None:
    rows = []
    for split in ("train", "dev", "test"):
        text = stats[split]["text"]
        caption = stats[split]["caption"]
        rows.append(
            f"{split} & {text['samples']} & {text['tokens']} & "
            f"{caption['tokens']} & {text['unique_images']} \\\\"
        )

    table = "\n".join(
        [
            "\\begin{tabular}{lrrrr}",
            "\\toprule",
            "数据划分 & 样本数 & Text token数 & Caption token数 & 图像数 \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )
    (OUT_DIR / "tables" / "dataset_statistics.tex").write_text(table + "\n", encoding="utf-8")


def write_label_table(stats: dict) -> None:
    domains = {"text": Counter(), "caption": Counter()}
    for split in stats.values():
        for domain in domains:
            domains[domain].update(split[domain]["labels"])

    labels = sorted(set(domains["text"]) | set(domains["caption"]))
    rows = [
        f"{latex_escape(label)} & {domains['text'].get(label, 0)} & {domains['caption'].get(label, 0)} \\\\"
        for label in labels
    ]
    table = "\n".join(
        [
            "\\begin{tabular}{lrr}",
            "\\toprule",
            "标签 & Text & Caption \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )
    (OUT_DIR / "tables" / "label_statistics.tex").write_text(table + "\n", encoding="utf-8")


def main() -> None:
    (OUT_DIR / "tables").mkdir(parents=True, exist_ok=True)
    stats = {
        split: {domain: parse_bio_file(path, domain) for domain, path in files.items()}
        for split, files in SPLITS.items()
    }
    (OUT_DIR / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_dataset_table(stats)
    write_label_table(stats)
    print(OUT_DIR / "dataset_stats.json")


if __name__ == "__main__":
    main()
