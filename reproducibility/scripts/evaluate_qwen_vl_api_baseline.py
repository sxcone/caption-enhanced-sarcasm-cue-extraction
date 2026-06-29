#!/usr/bin/env python3
"""Evaluate DashScope Qwen-VL outputs and create qwen_vl_max-named artifacts."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "paper_submission/results/vlm_api_baseline"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default=str(DEFAULT_OUT / "qwen_vl_max_raw_outputs.jsonl"))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator = ROOT / "paper_submission/scripts/evaluate_vlm_prompt_baseline.py"
    subprocess.run(
        ["python3", str(evaluator), "--predictions", args.predictions, "--output_dir", str(output_dir)],
        check=True,
    )
    mapping = {
        "vlm_baseline_summary.json": "qwen_vl_max_summary.json",
        "vlm_baseline_summary.md": "qwen_vl_max_summary.md",
        "vlm_baseline_table.tex": "qwen_vl_max_table.tex",
        "vlm_baseline_detailed.json": "qwen_vl_max_detailed.json",
    }
    for src_name, dst_name in mapping.items():
        src = output_dir / src_name
        if src.exists():
            shutil.copyfile(src, output_dir / dst_name)
    summary_md = output_dir / "qwen_vl_max_summary.md"
    if summary_md.exists():
        text = summary_md.read_text(encoding="utf-8")
        text = text.replace("# Local VLM prompt baseline", "# Qwen-VL-Max API prompt baseline")
        summary_md.write_text(text, encoding="utf-8")
    table_tex = output_dir / "qwen_vl_max_table.tex"
    if table_tex.exists():
        text = table_tex.read_text(encoding="utf-8")
        text = text.replace(
            "Local open-source VLM prompt baseline on the test split.",
            "Qwen-VL-Max API prompt baseline on the test split.",
        )
        table_tex.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
