#!/usr/bin/env python3
"""Evaluate a soft-vote ensemble of BERT sequence taggers."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


PAPER_DIR = Path(__file__).resolve().parents[1]
ROOT = PAPER_DIR.parent
CODE_DIR = ROOT / "SRTP" / "HVFormer-main" / "HVFormer-main"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from processor.mner_dataset import MNERDataset, MNERProcessor  # noqa: E402
from run import DATA_PATH  # noqa: E402
from scripts.train_ner_baselines import (  # noqa: E402
    BertBoundaryNER,
    BertLinearNER,
    build_pred_ids_argmax,
    collect_sequences,
    compute_detailed_metrics,
    compute_metrics,
    resolve_cached_model_source,
    set_seed,
)


RAW_DIR = PAPER_DIR / "results" / "mta_ensembles" / "raw"
TABLE_DIR = PAPER_DIR / "tables"
CKPT_DIR = CODE_DIR / "ckpt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="caption_ner", choices=["caption_ner", "text_ner"])
    parser.add_argument("--model_type", default="bert_linear", choices=["bert_linear", "bert_boundary"])
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--bert_name", default="bert-base-uncased")
    parser.add_argument("--max_seq", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--tune_decode_o_bias", action="store_true")
    parser.add_argument("--decode_o_bias_min", type=float, default=-1.5)
    parser.add_argument("--decode_o_bias_max", type=float, default=1.5)
    parser.add_argument("--decode_o_bias_step", type=float, default=0.25)
    parser.add_argument("--output_prefix", default="caption_bert_linear_ensemble")
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if device == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_processor(args: argparse.Namespace):
    label_alias_map = None
    if args.dataset_name == "caption_ner":
        label_alias_map = {
            "B-TENT": "B-MENT",
            "I-TENT": "I-MENT",
            "B-TOPI": "B-MOPI",
            "I-TOPI": "I-MOPI",
        }
    bert_source = resolve_cached_model_source(args.bert_name)
    processor = MNERProcessor(
        DATA_PATH[args.dataset_name],
        bert_source,
        collapse_bio=True,
        include_test_in_label_map=False,
        label_alias_map=label_alias_map,
    )
    return processor, bert_source


def load_models(args: argparse.Namespace, bert_source: str, num_labels: int, device: str):
    models = []
    model_cls = BertBoundaryNER if args.model_type == "bert_boundary" else BertLinearNER
    for ckpt_name in args.checkpoints:
        path = Path(ckpt_name)
        if not path.exists():
            path = CKPT_DIR / ckpt_name
        model = model_cls(bert_source, num_labels)
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        models.append(model)
    return models


def averaged_logits(models, input_ids, attention_mask, token_type_ids, model_type: str):
    logits = []
    with torch.no_grad():
        for model in models:
            if model_type == "bert_boundary":
                logits.append(model(input_ids, attention_mask, token_type_ids)["logits"])
            else:
                logits.append(model(input_ids, attention_mask, token_type_ids))
    return torch.stack(logits, dim=0).mean(dim=0)


def evaluate(models, loader, device, id2label, model_type: str, decode_bias=None):
    all_true, all_pred = [], []
    for batch in loader:
        batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)
        input_ids, token_type_ids, attention_mask, labels, eval_mask, *_ = batch
        logits = averaged_logits(models, input_ids, attention_mask, token_type_ids, model_type)
        pred_ids = build_pred_ids_argmax(logits, decode_bias=decode_bias)
        y_true, y_pred = collect_sequences(pred_ids, labels, eval_mask, attention_mask, id2label)
        all_true.extend(y_true)
        all_pred.extend(y_pred)
    metrics = compute_metrics(all_true, all_pred)
    metrics["detailed"] = compute_detailed_metrics(all_true, all_pred)
    return metrics


def tune_o_bias(args, models, loader, device, id2label, o_label_id):
    if not args.tune_decode_o_bias:
        return None, None
    num_labels = len(id2label)
    best_bias = None
    best_metrics = None
    for value in np.arange(args.decode_o_bias_min, args.decode_o_bias_max + 1e-9, args.decode_o_bias_step):
        bias = [0.0] * num_labels
        bias[o_label_id] = float(round(value, 4))
        metrics = evaluate(models, loader, device, id2label, args.model_type, decode_bias=bias)
        if best_metrics is None or metrics["f1"] > best_metrics["f1"]:
            best_bias = bias
            best_metrics = metrics
    return best_bias, best_metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    processor, bert_source = load_processor(args)
    dev_dataset = MNERDataset(processor, max_seq=args.max_seq, mode="dev")
    test_dataset = MNERDataset(processor, max_seq=args.max_seq, mode="test")
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    label_map = processor.get_label_map()
    id2label = {v: k for k, v in label_map.items()}
    o_label_id = label_map.get("O", 0)
    models = load_models(args, bert_source, len(label_map), device)

    decode_bias, dev_metrics = tune_o_bias(args, models, dev_loader, device, id2label, o_label_id)
    if dev_metrics is None:
        dev_metrics = evaluate(models, dev_loader, device, id2label, args.model_type)
    test_metrics = evaluate(models, test_loader, device, id2label, args.model_type, decode_bias=decode_bias)

    result = {
        "experiment_name": f"{args.output_prefix}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}",
        "dataset_name": args.dataset_name,
        "model_type": f"{args.model_type}_ensemble",
        "method_label": "BERT-Linear Ensemble" if args.model_type == "bert_linear" else "BC-CE-BERT Ensemble",
        "checkpoints": args.checkpoints,
        "device": device,
        "dev_metrics": dev_metrics,
        "final_test_metrics": test_metrics,
        "final_test_detailed_metrics": test_metrics.get("detailed", {}),
        "decode_bias": decode_bias,
        "tune_decode_o_bias": args.tune_decode_o_bias,
    }
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / f"{result['experiment_name']}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(out)
    print(json.dumps({
        "dev_f1": dev_metrics["f1"],
        "test_f1": test_metrics["f1"],
        "span_f1": test_metrics.get("detailed", {}).get("span", {}).get("f1"),
        "decode_bias": decode_bias,
    }, indent=2))


if __name__ == "__main__":
    main()
