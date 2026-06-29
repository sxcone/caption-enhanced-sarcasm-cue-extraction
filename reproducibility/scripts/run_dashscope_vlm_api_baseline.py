#!/usr/bin/env python3
"""Run a DashScope Qwen-VL API prompt baseline on the held-out test split."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import time
from pathlib import Path
from typing import Any

import requests
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
TEXT_FILE = ROOT / "caption和text" / "text.txt"
CAPTION_FILE = ROOT / "caption和text" / "caption.txt"
OUT_DIR = ROOT / "paper_submission" / "results" / "vlm_api_baseline"
IMAGE_DIRS = [
    ROOT / "SRTP" / "part1_img_580" / "part1_img_580",
    ROOT / "SRTP" / "part2_img_364",
    ROOT / "SRTP" / "part3_img_420",
    ROOT / "SRTP" / "part5_img_394",
    ROOT / "github_release" / "caption-enhanced-sarcasm-cue-extraction" / "part1_img_580",
    ROOT / "github_release" / "caption-enhanced-sarcasm-cue-extraction" / "part2_img_364",
    ROOT / "github_release" / "caption-enhanced-sarcasm-cue-extraction" / "part3_img_420",
    ROOT / "github_release" / "caption-enhanced-sarcasm-cue-extraction" / "part5_img_394",
]
SPLIT_SIZES = [("train", 1230), ("dev", 263), ("test", 265)]
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


def find_image(imgid: str) -> Path | None:
    for base in IMAGE_DIRS:
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            p = base / f"{imgid}{ext}"
            if p.exists():
                return p
    return None


def image_to_data_url(path: Path, max_side: int, jpeg_quality: int) -> str:
    if max_side <= 0:
        mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"
    from io import BytesIO

    image = Image.open(path).convert("RGB")
    image.thumbnail((max_side, max_side))
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def build_prompt(post_text: str, caption: str) -> str:
    schema = '{"TENT":[],"TOPI":[],"MENT":[],"MOPI":[],"rationale":""}'
    return (
        "You are evaluating a multimodal social-media post for sarcasm evidence. "
        "Extract only short cue phrases that are explicitly present in either the post text or the image. "
        "Return one compact JSON object only, with exactly these keys: TENT, TOPI, MENT, MOPI, rationale. "
        "TENT = textual sarcastic entity phrases copied from the post text. "
        "TOPI = textual opinion cue phrases copied from the post text. "
        "MENT = visual entity or anchor phrases grounded in the image or reference caption. "
        "MOPI = visual opinion or visual defect cue phrases grounded in the image or reference caption. "
        "Use empty arrays if no cue is clear. Keep phrases short and do not invent words that are not supported. "
        "Return at most 3 phrases for each cue type. Keep the rationale under 12 words. "
        "Do not include markdown, comments, explanations outside JSON, or alternative schemas.\n\n"
        f"Required JSON example: {schema}\n\n"
        f"Post text: {post_text}\n"
        f"Reference generated caption: {caption}\n"
        "JSON:"
    )


def extract_content(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return re.sub(r"\s+", " ", content).strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return re.sub(r"\s+", " ", " ".join(parts)).strip()
    return re.sub(r"\s+", " ", str(content)).strip()


def estimate_cost_usd(usage: dict[str, Any] | None, prompt: str, output: str, input_price: float, output_price: float) -> tuple[float, dict[str, int]]:
    usage = usage or {}
    input_tokens = int(
        usage.get("prompt_tokens")
        or usage.get("input_tokens")
        or usage.get("input_token_details", {}).get("text_tokens", 0)
        or max(1, len(prompt) // 4 + 1200)
    )
    output_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or max(1, len(output) // 4))
    cost = input_tokens * input_price / 1_000_000 + output_tokens * output_price / 1_000_000
    return cost, {"input_tokens": input_tokens, "output_tokens": output_tokens}


def load_samples(limit: int | None) -> list[dict[str, str]]:
    text_samples = test_split(parse_bio(TEXT_FILE))
    caption_samples = test_split(parse_bio(CAPTION_FILE))
    samples = []
    for ts, cs in zip(text_samples, caption_samples):
        img_path = find_image(ts["imgid"])
        samples.append(
            {
                "imgid": ts["imgid"],
                "post_text": " ".join(ts["tokens"]),
                "caption": " ".join(cs["tokens"]),
                "image_path": str(img_path) if img_path else "",
            }
        )
    return samples[:limit] if limit else samples


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            done.add(json.loads(line)["imgid"])
        except Exception:
            continue
    return done


def call_dashscope(
    endpoint: str,
    api_key: str,
    model: str,
    sample: dict[str, str],
    max_tokens: int,
    timeout: int,
    max_image_side: int,
    jpeg_quality: int,
) -> dict[str, Any]:
    prompt = build_prompt(sample["post_text"], sample["caption"])
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_data_url(Path(sample["image_path"]), max_image_side, jpeg_quality)},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    response = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
    if response.status_code >= 400:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:1000]}")
    data = response.json()
    output = extract_content(data)
    return {"raw_response": data, "raw_prediction": output, "prompt": prompt}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen-vl-max")
    parser.add_argument("--endpoint", default="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--max_image_side", type=int, default=768)
    parser.add_argument("--jpeg_quality", type=int, default=85)
    parser.add_argument("--budget_usd", type=float, default=5.0)
    parser.add_argument("--input_price_usd_per_million", type=float, default=0.23)
    parser.add_argument("--output_price_usd_per_million", type=float, default=0.574)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--output", default=str(OUT_DIR / "qwen_vl_max_raw_outputs.jsonl"))
    parser.add_argument("--errors", default=str(OUT_DIR / "qwen_vl_max_errors.log"))
    parser.add_argument("--run_config", default=str(OUT_DIR / "qwen_vl_max_run_config.json"))
    args = parser.parse_args()

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise SystemExit(
            "DASHSCOPE_API_KEY is not set. Configure it first, for example:\n"
            '  export DASHSCOPE_API_KEY="your_dashscope_api_key"\n'
            "No API request was sent and no result file was generated."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output)
    err_path = Path(args.errors)
    config_path = Path(args.run_config)
    samples = load_samples(args.limit)
    done = completed_ids(out_path)
    spent = 0.0
    processed = 0
    failed = 0
    token_usage = {"input_tokens": 0, "output_tokens": 0}
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    config = {
        "model": args.model,
        "endpoint": args.endpoint,
        "limit": args.limit,
        "sample_count": len(samples),
        "max_tokens": args.max_tokens,
        "max_image_side": args.max_image_side,
        "jpeg_quality": args.jpeg_quality,
        "budget_usd": args.budget_usd,
        "input_price_usd_per_million": args.input_price_usd_per_million,
        "output_price_usd_per_million": args.output_price_usd_per_million,
        "started_at": started_at,
        "note": "DashScope Qwen-VL API zero-shot prompt diagnostic. Cost is estimated from returned or fallback token counts.",
    }
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    with out_path.open("a", encoding="utf-8") as out, err_path.open("a", encoding="utf-8") as err:
        for idx, sample in enumerate(samples, start=1):
            if sample["imgid"] in done:
                continue
            if spent >= args.budget_usd:
                err.write(json.dumps({"event": "budget_stop", "estimated_cost_usd": spent}, ensure_ascii=False) + "\n")
                break
            if not sample["image_path"]:
                failed += 1
                err.write(json.dumps({"imgid": sample["imgid"], "error": "missing image"}, ensure_ascii=False) + "\n")
                continue
            try:
                result = call_dashscope(
                    args.endpoint,
                    api_key,
                    args.model,
                    sample,
                    args.max_tokens,
                    args.timeout,
                    args.max_image_side,
                    args.jpeg_quality,
                )
                cost, usage = estimate_cost_usd(
                    result.get("raw_response", {}).get("usage"),
                    result["prompt"],
                    result["raw_prediction"],
                    args.input_price_usd_per_million,
                    args.output_price_usd_per_million,
                )
                spent += cost
                token_usage["input_tokens"] += usage["input_tokens"]
                token_usage["output_tokens"] += usage["output_tokens"]
                row = {
                    **sample,
                    "model": args.model,
                    "provider": "DashScope",
                    "raw_prediction": result["raw_prediction"],
                    "usage": result.get("raw_response", {}).get("usage", usage),
                    "estimated_cost_usd": cost,
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                processed += 1
                print(f"[{idx}/{len(samples)}] {sample['imgid']} ok cost~${cost:.5f} total~${spent:.4f}", flush=True)
            except Exception as exc:
                failed += 1
                err.write(
                    json.dumps(
                        {"imgid": sample["imgid"], "error": f"{type(exc).__name__}: {exc}"},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                err.flush()
                print(f"[{idx}/{len(samples)}] {sample['imgid']} failed: {type(exc).__name__}: {exc}", flush=True)

    final_config = {
        **config,
        "completed_new_samples": processed,
        "failed_new_samples": failed,
        "estimated_cost_usd": spent,
        "estimated_token_usage": token_usage,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    config_path.write_text(json.dumps(final_config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(final_config, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
