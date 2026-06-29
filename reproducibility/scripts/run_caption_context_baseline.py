#!/usr/bin/env python3
"""Run BERT sequence-labelling baselines with cross-channel caption/text context.

This script builds a lightweight comparison inspired by caption-enhanced and
cross-modal semantic fusion papers. For a primary channel, the other channel is
appended as context after a separator, while loss and evaluation are computed
only on the primary-channel tokens.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup


PAPER_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = PAPER_DIR.parents[0]
HVFORMER_DIR = PROJECT_DIR / "SRTP" / "HVFormer-main" / "HVFormer-main"
CACHE_DIR = HVFORMER_DIR / ".cache" / "huggingface"
DATA_DIR = PROJECT_DIR / "caption和text"
OUT_DIR = PAPER_DIR / "results" / "contrast" / "raw"

os.environ.setdefault("HF_HOME", str(CACHE_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(CACHE_DIR))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

SPLIT_FILES = {
    "text": {
        "train": DATA_DIR / "text_part1_1000_580.txt",
        "dev": DATA_DIR / "text_part2_650_364.txt",
        "test": DATA_DIR / "text_3545_part5_595_394.txt",
    },
    "caption": {
        "train": DATA_DIR / "caption_3545_part1_1000_580.txt",
        "dev": DATA_DIR / "caption_part2_650_364.txt",
        "test": DATA_DIR / "caption_3545_part5_595_394.txt",
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> str:
    if device in {"cpu", "cuda", "mps"}:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_cached_model(model_name: str) -> str:
    if Path(model_name).exists():
        return model_name
    repo_name = model_name.replace("/", "--")
    snapshots_root = CACHE_DIR / "hub" / f"models--{repo_name}" / "snapshots"
    if not snapshots_root.is_dir():
        return model_name
    usable = []
    for snap in snapshots_root.iterdir():
        if (snap / "config.json").is_file() and (
            (snap / "pytorch_model.bin").is_file() or (snap / "model.safetensors").is_file()
        ):
            usable.append(str(snap))
    return sorted(usable)[-1] if usable else model_name


def collapse_label(label: str) -> str:
    aliases = {"OS": "O", "N-MOPI": "I-MOPI"}
    label = aliases.get(label, label)
    if label == "O":
        return "O"
    if "-" in label:
        return label.split("-", 1)[1]
    return label


@dataclass
class Sample:
    imgid: str
    tokens: list[str]
    labels: list[str]


def read_bio_file(path: Path) -> dict[str, Sample]:
    samples: dict[str, Sample] = {}
    cur_imgid: str | None = None
    tokens: list[str] = []
    labels: list[str] = []

    def flush() -> None:
        nonlocal cur_imgid, tokens, labels
        if cur_imgid is not None and tokens:
            samples[cur_imgid] = Sample(cur_imgid, tokens, labels)
        cur_imgid = None
        tokens = []
        labels = []

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            flush()
            continue
        if line.startswith("IMGID:"):
            flush()
            cur_imgid = line.split(":", 1)[1].strip()
            continue
        parts = line.split("\t")
        if len(parts) == 2:
            tokens.append(parts[0])
            labels.append(collapse_label(parts[1]))
    flush()
    return samples


def build_label_map(primary: str) -> dict[str, int]:
    labels = {"O"}
    for split in ("train", "dev"):
        for sample in read_bio_file(SPLIT_FILES[primary][split]).values():
            labels.update(sample.labels)
    ordered = ["O"] + sorted(label for label in labels if label != "O")
    return {label: idx for idx, label in enumerate(ordered)}


class ContextNERDataset(Dataset):
    def __init__(
        self,
        primary: str,
        context: str,
        split: str,
        tokenizer: BertTokenizer,
        label_map: dict[str, int],
        max_seq: int,
    ) -> None:
        self.primary = primary
        self.context = context
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_seq = max_seq

        primary_samples = read_bio_file(SPLIT_FILES[primary][split])
        context_samples = read_bio_file(SPLIT_FILES[context][split])
        shared = [imgid for imgid in primary_samples if imgid in context_samples]
        self.samples = [(primary_samples[imgid], context_samples[imgid]) for imgid in shared]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        primary, context = self.samples[idx]
        bert_tokens = [self.tokenizer.cls_token]
        labels = [-100]
        eval_mask = [0]
        token_type_ids = [0]

        for token, label in zip(primary.tokens, primary.labels):
            pieces = self.tokenizer.tokenize(token) or [self.tokenizer.unk_token]
            for j, piece in enumerate(pieces):
                if len(bert_tokens) >= self.max_seq - 2:
                    break
                bert_tokens.append(piece)
                labels.append(self.label_map.get(label, self.label_map["O"]) if j == 0 else -100)
                eval_mask.append(1 if j == 0 else 0)
                token_type_ids.append(0)
            if len(bert_tokens) >= self.max_seq - 2:
                break

        bert_tokens.append(self.tokenizer.sep_token)
        labels.append(-100)
        eval_mask.append(0)
        token_type_ids.append(0)

        for token in context.tokens:
            pieces = self.tokenizer.tokenize(token) or [self.tokenizer.unk_token]
            for piece in pieces:
                if len(bert_tokens) >= self.max_seq - 1:
                    break
                bert_tokens.append(piece)
                labels.append(-100)
                eval_mask.append(0)
                token_type_ids.append(1)
            if len(bert_tokens) >= self.max_seq - 1:
                break

        bert_tokens.append(self.tokenizer.sep_token)
        labels.append(-100)
        eval_mask.append(0)
        token_type_ids.append(1)

        input_ids = self.tokenizer.convert_tokens_to_ids(bert_tokens)
        attention_mask = [1] * len(input_ids)
        pad_len = self.max_seq - len(input_ids)
        if pad_len > 0:
            input_ids += [self.tokenizer.pad_token_id] * pad_len
            attention_mask += [0] * pad_len
            token_type_ids += [0] * pad_len
            labels += [-100] * pad_len
            eval_mask += [0] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "eval_mask": torch.tensor(eval_mask, dtype=torch.long),
        }


class BertContextNER(nn.Module):
    def __init__(self, bert_name: str, num_labels: int) -> None:
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        return self.classifier(self.dropout(output.last_hidden_state))


def compute_class_weights(dataset: ContextNERDataset, label_map: dict[str, int], power: float, o_weight: float) -> torch.Tensor:
    counts = np.ones(len(label_map), dtype=np.float64)
    for item in dataset:
        for label_id in item["labels"].tolist():
            if label_id >= 0:
                counts[label_id] += 1.0
    inv = (counts.sum() / counts) ** power
    inv = inv / inv.mean()
    inv[label_map["O"]] *= o_weight
    return torch.tensor(inv, dtype=torch.float32)


def focal_cross_entropy(logits, labels, class_weights, gamma: float):
    active = labels.view(-1) != -100
    active_logits = logits.view(-1, logits.size(-1))[active]
    active_labels = labels.view(-1)[active]
    ce = nn.functional.cross_entropy(active_logits, active_labels, weight=class_weights, reduction="none")
    if gamma <= 0:
        return ce.mean()
    pt = torch.exp(-ce)
    return (((1 - pt) ** gamma) * ce).mean()


def evaluate(model, loader, device, id2label):
    model.eval()
    y_true: list[str] = []
    y_pred: list[str] = []
    total_loss = 0.0
    steps = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["input_ids"], batch["attention_mask"], batch["token_type_ids"])
            preds = logits.argmax(dim=-1)
            active = batch["eval_mask"] == 1
            true_ids = batch["labels"][active].detach().cpu().tolist()
            pred_ids = preds[active].detach().cpu().tolist()
            y_true.extend(id2label.get(idx, "O") for idx in true_ids)
            y_pred.extend(id2label.get(idx, "O") for idx in pred_ids)
            steps += 1

    eval_labels = sorted(label for label in set(y_true + y_pred) if label != "O")
    if not eval_labels:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "report": "No non-O labels."}
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=eval_labels, average="micro", zero_division=0
    )
    return {
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100,
        "loss": total_loss / max(steps, 1),
        "report": classification_report(y_true, y_pred, labels=eval_labels, digits=4, zero_division=0),
    }


def run_one(args, primary: str, context: str) -> Path:
    set_seed(args.seed)
    device = resolve_device(args.device)
    bert_source = resolve_cached_model(args.bert_name)
    tokenizer = BertTokenizer.from_pretrained(bert_source, do_lower_case=True)
    label_map = build_label_map(primary)
    id2label = {idx: label for label, idx in label_map.items()}

    datasets = {
        split: ContextNERDataset(primary, context, split, tokenizer, label_map, args.max_seq)
        for split in ("train", "dev", "test")
    }
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True),
        "dev": DataLoader(datasets["dev"], batch_size=args.batch_size, shuffle=False),
        "test": DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False),
    }

    model = BertContextNER(bert_source, len(label_map)).to(device)
    class_weights = compute_class_weights(datasets["train"], label_map, args.class_weight_power, args.o_loss_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(loaders["train"]) * args.num_epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=math.ceil(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    best_dev = {"f1": -1.0}
    best_test = {"f1": 0.0}
    history = []

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(loaders["train"], start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["input_ids"], batch["attention_mask"], batch["token_type_ids"])
            loss = focal_cross_entropy(logits, batch["labels"], class_weights, args.focal_gamma)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running_loss += float(loss.detach().cpu().item())
            if args.log_steps and step % args.log_steps == 0:
                print(f"{primary}+{context} seed={args.seed} epoch={epoch} step={step}/{len(loaders['train'])} loss={running_loss / step:.5f}")

        dev_metrics = evaluate(model, loaders["dev"], device, id2label)
        test_metrics = evaluate(model, loaders["test"], device, id2label)
        history.append({"epoch": epoch, "dev": dev_metrics, "test": test_metrics})
        print(
            f"{primary}+{context} seed={args.seed} epoch={epoch}: "
            f"dev_f1={dev_metrics['f1']:.4f}, test_f1={test_metrics['f1']:.4f}"
        )
        if dev_metrics["f1"] > best_dev["f1"]:
            best_dev = dev_metrics
            best_test = test_metrics

    final_test = evaluate(model, loaders["test"], device, id2label)
    stamp = time.strftime("%Y_%m_%d_%H_%M_%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"bert_context_{primary}_with_{context}_seed{args.seed}_{stamp}.json"
    result = {
        "model": "BERT-Context",
        "primary": primary,
        "context": context,
        "label": f"BERT-{primary.title()}+{context.title()}Context",
        "seed": args.seed,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "max_seq": args.max_seq,
        "device": device,
        "label_map": label_map,
        "dataset_sizes": {split: len(ds) for split, ds in datasets.items()},
        "best_dev_metrics": best_dev,
        "test_at_best_dev": best_test,
        "final_test_metrics": final_test,
        "history": history,
    }
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", choices=["text", "caption", "both"], default="both")
    parser.add_argument("--bert_name", default="bert-base-uncased")
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_seq", type=int, default=128)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--o_loss_weight", type=float, default=0.2)
    parser.add_argument("--class_weight_power", type=float, default=0.7)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--log_steps", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = []
    if args.primary in {"text", "both"}:
        tasks.append(("text", "caption"))
    if args.primary in {"caption", "both"}:
        tasks.append(("caption", "text"))
    for primary, context in tasks:
        out_path = run_one(args, primary, context)
        print(out_path)


if __name__ == "__main__":
    main()
