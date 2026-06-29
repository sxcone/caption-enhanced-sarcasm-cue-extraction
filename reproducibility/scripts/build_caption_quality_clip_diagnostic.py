from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


ROOT = Path("/Users/sxc/Desktop/去年的SRTP")
PAPER = ROOT / "paper_submission"
DATA_DIR = ROOT / "caption和text"
HVFORMER = ROOT / "SRTP" / "HVFormer-main" / "HVFormer-main"
CACHE_DIR = HVFORMER / ".cache" / "huggingface"
OUT_DIR = PAPER / "results" / "caption_quality"
CLIP_OUT_DIR = PAPER / "results" / "clip_diagnostic"
FIG_DIR = PAPER / "multimedia_systems_submission" / "figures"

os.environ.setdefault("HF_HOME", str(CACHE_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(CACHE_DIR))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

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

IMAGE_DIRS = [
    ROOT / "SRTP" / "part1_img_580" / "part1_img_580",
    ROOT / "SRTP" / "part2_img_364",
    ROOT / "SRTP" / "part3_img_420",
    ROOT / "SRTP" / "part5_img_394",
    ROOT / "github_release" / "caption-enhanced-sarcasm-cue-extraction" / "part1_img_580" / "part1_img_580",
    ROOT / "github_release" / "caption-enhanced-sarcasm-cue-extraction" / "part2_img_364",
    ROOT / "github_release" / "caption-enhanced-sarcasm-cue-extraction" / "part3_img_420",
    ROOT / "github_release" / "caption-enhanced-sarcasm-cue-extraction" / "part5_img_394",
]

LABEL_ALIASES = {"OS": "O", "N-MOPI": "I-MOPI"}
STOP = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "with",
    "for",
    "is",
    "are",
    "this",
    "that",
    "it",
    "as",
    "at",
    "by",
}


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)

PALETTE = {
    "text": "#4C78A8",
    "caption": "#F58518",
    "clip": "#54A24B",
    "accent": "#7E57C2",
    "ink": "#263238",
    "muted": "#607D8B",
    "grid": "#CFD8DC",
}


def normalise_label(label: str) -> str:
    return LABEL_ALIASES.get(label, label)


def read_bio(path: Path) -> dict[str, dict]:
    samples: dict[str, dict] = {}
    imgid = None
    tokens: list[str] = []
    labels: list[str] = []

    def flush() -> None:
        nonlocal imgid, tokens, labels
        if imgid is not None:
            samples[imgid] = {"imgid": imgid, "tokens": tokens, "labels": labels}
        imgid = None
        tokens = []
        labels = []

    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            flush()
            continue
        if line.startswith("IMGID:"):
            flush()
            imgid = line.split(":", 1)[1].strip()
            continue
        if "\t" not in raw:
            parts = raw.rsplit(None, 1)
        else:
            parts = raw.rsplit("\t", 1)
        if len(parts) != 2:
            continue
        token, label = parts[0], normalise_label(parts[1].strip())
        if label != "O" and not (label.startswith("B-") or label.startswith("I-")):
            continue
        tokens.append(token)
        labels.append(label)
    flush()
    return samples


def spans(labels: list[str]) -> list[tuple[str, int, int]]:
    out: list[tuple[str, int, int]] = []
    cur = None
    start = 0
    for i, lab in enumerate(labels):
        if lab == "O":
            if cur is not None:
                out.append((cur, start, i - 1))
                cur = None
            continue
        prefix, typ = lab.split("-", 1)
        if prefix == "B" or cur != typ:
            if cur is not None:
                out.append((cur, start, i - 1))
            cur = typ
            start = i
    if cur is not None:
        out.append((cur, start, len(labels) - 1))
    return out


def words(tokens: list[str]) -> set[str]:
    text = " ".join(tokens).lower()
    out = set(re.findall(r"[a-z][a-z0-9_-]+", text))
    return {w for w in out if w not in STOP and len(w) > 2 and not w.startswith("emoji")}


def find_image(imgid: str) -> Path | None:
    for directory in IMAGE_DIRS:
        for ext in (".jpg", ".jpeg", ".png"):
            path = directory / f"{imgid}{ext}"
            if path.exists():
                return path
    return None


def row_for(split: str, imgid: str, text_sample: dict, cap_sample: dict) -> dict:
    text_tokens = text_sample["tokens"]
    cap_tokens = cap_sample["tokens"]
    text_labels = text_sample["labels"]
    cap_labels = cap_sample["labels"]
    cap_spans = spans(cap_labels)
    text_spans = spans(text_labels)
    cap_pos = sum(1 for label in cap_labels if label != "O")
    text_pos = sum(1 for label in text_labels if label != "O")
    cap_words = words(cap_tokens)
    text_words = words(text_tokens)
    inter = cap_words & text_words
    union = cap_words | text_words
    label_counts = Counter(label.split("-", 1)[1] for label in cap_labels if label != "O" and "-" in label)
    return {
        "split": split,
        "imgid": imgid,
        "text": " ".join(text_tokens),
        "caption": " ".join(cap_tokens),
        "text_len": len(text_tokens),
        "caption_len": len(cap_tokens),
        "text_positive_tokens": text_pos,
        "caption_positive_tokens": cap_pos,
        "text_cue_density": text_pos / max(1, len(text_tokens)),
        "caption_cue_density": cap_pos / max(1, len(cap_tokens)),
        "text_span_count": len(text_spans),
        "caption_span_count": len(cap_spans),
        "caption_has_ment": int(label_counts.get("MENT", 0) > 0),
        "caption_has_mopi": int(label_counts.get("MOPI", 0) > 0),
        "caption_ment_tokens": label_counts.get("MENT", 0),
        "caption_mopi_tokens": label_counts.get("MOPI", 0),
        "lexical_jaccard": len(inter) / max(1, len(union)),
        "image_path": str(find_image(imgid) or ""),
    }


def build_rows() -> list[dict]:
    rows = []
    for split, paths in SPLITS.items():
        text = read_bio(paths["text"])
        caption = read_bio(paths["caption"])
        for imgid in sorted(set(text) & set(caption), key=lambda x: int(x) if x.isdigit() else x):
            rows.append(row_for(split, imgid, text[imgid], caption[imgid]))
    return rows


def resolve_cached_model(model_name: str) -> str:
    repo = model_name.replace("/", "--")
    root = CACHE_DIR / "hub" / f"models--{repo}" / "snapshots"
    if not root.exists():
        return model_name
    usable = sorted(
        p
        for p in root.iterdir()
        if (p / "config.json").exists()
        and ((p / "preprocessor_config.json").exists() or (p / "preprocessor_config.json").is_symlink())
    )
    return str(usable[-1]) if usable else model_name


def compute_clip(rows: list[dict], split: str, batch_size: int) -> list[dict]:
    selected = [row for row in rows if row["split"] == split and row["image_path"]]
    if not selected:
        return []
    device = "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    model_name = resolve_cached_model("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(selected), batch_size):
            batch = selected[i : i + batch_size]
            images = [Image.open(row["image_path"]).convert("RGB") for row in batch]
            cap_inputs = processor(text=[row["caption"] for row in batch], images=images, return_tensors="pt", padding=True, truncation=True)
            text_inputs = processor(text=[row["text"] for row in batch], images=images, return_tensors="pt", padding=True, truncation=True)
            cap_inputs = {k: v.to(device) for k, v in cap_inputs.items()}
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            cap_outputs = model(**cap_inputs)
            text_outputs = model(**text_inputs)
            img_feat = cap_outputs.image_embeds / cap_outputs.image_embeds.norm(dim=-1, keepdim=True)
            cap_feat = cap_outputs.text_embeds / cap_outputs.text_embeds.norm(dim=-1, keepdim=True)
            text_feat = text_outputs.text_embeds / text_outputs.text_embeds.norm(dim=-1, keepdim=True)
            cap_sim = (img_feat * cap_feat).sum(dim=-1).detach().cpu().numpy()
            text_sim = (img_feat * text_feat).sum(dim=-1).detach().cpu().numpy()
            for row, c, t in zip(batch, cap_sim, text_sim):
                item = dict(row)
                item["clip_image_caption"] = float(c)
                item["clip_image_text"] = float(t)
                item["clip_caption_margin"] = float(c - t)
                out.append(item)
    return out


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarise(rows: list[dict], clip_rows: list[dict]) -> dict:
    def mean(key: str, source: list[dict] = rows) -> float:
        vals = [float(row[key]) for row in source if row.get(key) not in ("", None)]
        return float(np.mean(vals)) if vals else 0.0

    summary = {
        "num_samples": len(rows),
        "splits": dict(Counter(row["split"] for row in rows)),
        "caption_length_mean": mean("caption_len"),
        "caption_cue_density_mean": mean("caption_cue_density"),
        "text_cue_density_mean": mean("text_cue_density"),
        "caption_span_count_mean": mean("caption_span_count"),
        "lexical_jaccard_mean": mean("lexical_jaccard"),
        "caption_has_ment_rate": mean("caption_has_ment"),
        "caption_has_mopi_rate": mean("caption_has_mopi"),
    }
    if clip_rows:
        summary.update(
            {
                "clip_split": clip_rows[0]["split"],
                "clip_samples": len(clip_rows),
                "clip_image_caption_mean": mean("clip_image_caption", clip_rows),
                "clip_image_text_mean": mean("clip_image_text", clip_rows),
                "clip_caption_margin_mean": mean("clip_caption_margin", clip_rows),
                "clip_caption_margin_positive_rate": float(np.mean([row["clip_caption_margin"] > 0 for row in clip_rows])),
            }
        )
    return summary


def save_figure(rows: list[dict], clip_rows: list[dict]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.1, 4.1))
    grid = fig.add_gridspec(2, 3, wspace=0.42, hspace=0.55)
    fig.suptitle("Caption quality and image-text alignment diagnostics", x=0.02, y=0.99, ha="left", fontsize=10, fontweight="bold")

    ax1 = fig.add_subplot(grid[0, 0])
    ax1.hist([row["caption_len"] for row in rows], bins=24, color=PALETTE["caption"], alpha=0.86)
    ax1.set_title("Caption length", loc="left", fontsize=8.5, fontweight="bold")
    ax1.set_xlabel("Tokens")
    ax1.set_ylabel("Samples")
    ax1.text(-0.18, 1.08, "a", transform=ax1.transAxes, fontsize=10, fontweight="bold")

    ax2 = fig.add_subplot(grid[0, 1])
    ax2.scatter(
        [row["caption_len"] for row in rows],
        [100 * row["caption_cue_density"] for row in rows],
        s=8,
        alpha=0.35,
        color=PALETTE["accent"],
        edgecolors="none",
    )
    ax2.set_title("Cue density vs. length", loc="left", fontsize=8.5, fontweight="bold")
    ax2.set_xlabel("Caption tokens")
    ax2.set_ylabel("Positive tokens (%)")
    ax2.text(-0.18, 1.08, "b", transform=ax2.transAxes, fontsize=10, fontweight="bold")

    ax3 = fig.add_subplot(grid[0, 2])
    rates = [np.mean([row["caption_has_ment"] for row in rows]), np.mean([row["caption_has_mopi"] for row in rows])]
    bars = ax3.bar(["MENT", "MOPI"], [100 * r for r in rates], color=[PALETTE["clip"], PALETTE["caption"]], width=0.56)
    ax3.set_ylim(0, 100)
    ax3.set_ylabel("Samples with cue (%)")
    ax3.set_title("Caption cue coverage", loc="left", fontsize=8.5, fontweight="bold")
    for bar, rate in zip(bars, rates):
        ax3.text(bar.get_x() + bar.get_width() / 2, 100 * rate + 2, f"{100*rate:.1f}%", ha="center", fontsize=6.5)
    ax3.text(-0.18, 1.08, "c", transform=ax3.transAxes, fontsize=10, fontweight="bold")

    ax4 = fig.add_subplot(grid[1, 0])
    ax4.hist([row["lexical_jaccard"] for row in rows], bins=22, color=PALETTE["text"], alpha=0.86)
    ax4.set_title("Text-caption lexical overlap", loc="left", fontsize=8.5, fontweight="bold")
    ax4.set_xlabel("Jaccard")
    ax4.set_ylabel("Samples")
    ax4.text(-0.18, 1.08, "d", transform=ax4.transAxes, fontsize=10, fontweight="bold")

    ax5 = fig.add_subplot(grid[1, 1:])
    if clip_rows:
        cap = [row["clip_image_caption"] for row in clip_rows]
        txt = [row["clip_image_text"] for row in clip_rows]
        ax5.boxplot([cap, txt], labels=["Image-caption", "Image-text"], widths=0.46, patch_artist=True)
        colors = [PALETTE["caption"], PALETTE["text"]]
        for patch, color in zip(ax5.artists, colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        ax5.scatter(np.random.normal(1, 0.03, len(cap)), cap, s=7, color=PALETTE["caption"], alpha=0.28, edgecolors="none")
        ax5.scatter(np.random.normal(2, 0.03, len(txt)), txt, s=7, color=PALETTE["text"], alpha=0.28, edgecolors="none")
        margin = np.mean([row["clip_caption_margin"] for row in clip_rows])
        ax5.text(0.02, 0.92, f"Mean margin = {margin:.3f}", transform=ax5.transAxes, fontsize=7, color=PALETTE["muted"])
    else:
        ax5.text(0.5, 0.5, "CLIP diagnostic unavailable", ha="center", va="center", color=PALETTE["muted"])
    ax5.set_title("CLIP visual alignment diagnostic", loc="left", fontsize=8.5, fontweight="bold")
    ax5.set_ylabel("Cosine similarity")
    ax5.text(-0.08, 1.08, "e", transform=ax5.transAxes, fontsize=10, fontweight="bold")

    fig.text(
        0.02,
        0.01,
        "Source data: BIO caption/text files and local images. CLIP scores are diagnostic alignment measures, not supervised cue-extraction F1.",
        fontsize=5.8,
        color=PALETTE["muted"],
    )
    for ext in ("pdf", "svg"):
        fig.savefig(FIG_DIR / f"Fig6.{ext}", bbox_inches="tight")
    fig.savefig(FIG_DIR / "Fig6.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--skip_clip", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CLIP_OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    clip_rows: list[dict] = []
    if not args.skip_clip:
        try:
            clip_rows = compute_clip(rows, args.clip_split, args.batch_size)
        except Exception as exc:
            (CLIP_OUT_DIR / "clip_error.log").write_text(repr(exc) + "\n", encoding="utf-8")
            clip_rows = []
    summary = summarise(rows, clip_rows)
    save_csv(OUT_DIR / "caption_quality_rows.csv", rows)
    save_csv(CLIP_OUT_DIR / f"clip_alignment_{args.clip_split}.csv", clip_rows)
    (OUT_DIR / "caption_quality_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md = [
        "# Caption quality and CLIP diagnostic summary",
        "",
        f"- Samples: {summary['num_samples']}",
        f"- Mean caption length: {summary['caption_length_mean']:.2f} tokens",
        f"- Mean caption cue density: {100*summary['caption_cue_density_mean']:.2f}%",
        f"- Mean text cue density: {100*summary['text_cue_density_mean']:.2f}%",
        f"- Mean text-caption lexical Jaccard: {summary['lexical_jaccard_mean']:.4f}",
        f"- Caption samples with MENT: {100*summary['caption_has_ment_rate']:.2f}%",
        f"- Caption samples with MOPI: {100*summary['caption_has_mopi_rate']:.2f}%",
    ]
    if clip_rows:
        md.extend(
            [
                f"- CLIP split: {summary['clip_split']} ({summary['clip_samples']} samples)",
                f"- Mean CLIP image-caption similarity: {summary['clip_image_caption_mean']:.4f}",
                f"- Mean CLIP image-text similarity: {summary['clip_image_text_mean']:.4f}",
                f"- Mean image-caption margin: {summary['clip_caption_margin_mean']:.4f}",
                f"- Positive caption margin rate: {100*summary['clip_caption_margin_positive_rate']:.2f}%",
            ]
        )
    else:
        md.append("- CLIP diagnostic unavailable; see clip_error.log if present.")
    (OUT_DIR / "caption_quality_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    save_figure(rows, clip_rows)
    print(OUT_DIR / "caption_quality_summary.json")
    print(FIG_DIR / "Fig6.pdf")


if __name__ == "__main__":
    main()
