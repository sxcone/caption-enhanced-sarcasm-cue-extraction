import argparse
import copy
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from sklearn.metrics import classification_report as sk_classification_report
from sklearn.metrics import precision_recall_fscore_support
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import AutoModel, get_linear_schedule_with_warmup

try:
    from torchcrf import CRF as _CRF
    _CRF_IS_TORCHCRF = True
except ImportError:
    from TorchCRF import CRF as _CRF
    _CRF_IS_TORCHCRF = False

from processor.mner_dataset import MNERProcessor, MNERDataset
from run import DATA_PATH, CAPTION_TEXT_DIR

BASE_DIR = Path('/Users/sxc/Desktop/去年的SRTP/SRTP/HVFormer-main/HVFormer-main')
CACHE_DIR = BASE_DIR / '.cache' / 'huggingface'
os.environ.setdefault('HF_HOME', str(CACHE_DIR))
os.environ.setdefault('TRANSFORMERS_CACHE', str(CACHE_DIR))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_cached_model_source(model_name_or_path: str) -> str:
    if os.path.exists(model_name_or_path):
        return model_name_or_path
    repo_name = model_name_or_path.replace('/', '--')
    snapshots_root = CACHE_DIR / 'hub' / f'models--{repo_name}' / 'snapshots'
    if not snapshots_root.is_dir():
        return model_name_or_path
    usable = []
    for snap_dir in snapshots_root.iterdir():
        cfg = snap_dir / 'config.json'
        pt = snap_dir / 'pytorch_model.bin'
        sf = snap_dir / 'model.safetensors'
        if cfg.is_file() and (pt.is_file() or sf.is_file()):
            usable.append(str(snap_dir))
    return sorted(usable)[-1] if usable else model_name_or_path


def resolve_device(device_arg: str) -> str:
    if device_arg in ('cuda', 'cpu', 'mps'):
        return device_arg
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


class BertLinearNER(nn.Module):
    def __init__(self, bert_name: str, num_labels: int):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        try:
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_dict=True,
            )
        except TypeError:
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
        x = self.dropout(outputs.last_hidden_state)
        return self.classifier(x)


class BertBoundaryNER(nn.Module):
    def __init__(self, bert_name: str, num_labels: int):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_name)
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden, num_labels)
        self.start_head = nn.Linear(hidden, 2)
        self.end_head = nn.Linear(hidden, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        try:
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_dict=True,
            )
        except TypeError:
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
        x = self.dropout(outputs.last_hidden_state)
        return {
            'logits': self.classifier(x),
            'start_logits': self.start_head(x),
            'end_logits': self.end_head(x),
        }


class BertCRFNER(nn.Module):
    def __init__(self, bert_name: str, num_labels: int):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        if _CRF_IS_TORCHCRF:
            self.crf = _CRF(num_labels, batch_first=True)
            self._crf_style = 'torchcrf'
        else:
            use_gpu = False
            self.crf = _CRF(num_labels, pad_idx=None, use_gpu=use_gpu)
            self._crf_style = 'TorchCRF'

    def emissions(self, input_ids, attention_mask, token_type_ids):
        try:
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_dict=True,
            )
        except TypeError:
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
        x = self.dropout(outputs.last_hidden_state)
        return self.classifier(x)

    def crf_loss(self, emissions, labels, mask):
        if self._crf_style == 'torchcrf':
            return -1 * self.crf(emissions, labels, mask=mask, reduction='mean')
        llh = self.crf(emissions, labels, mask)
        if isinstance(llh, torch.Tensor) and llh.dim() > 0:
            llh = llh.mean()
        return -1 * llh

    def decode(self, emissions, mask):
        if hasattr(self.crf, 'decode'):
            return self.crf.decode(emissions, mask)
        return self.crf.viterbi_decode(emissions, mask)


class PairedContextMNERDataset(MNERDataset):
    """Append paired text/caption as non-evaluated context for dataset-driven cue extraction."""

    def __init__(self, processor, companion_processor, max_seq=128, context_max_tokens=48, mode='train'):
        super().__init__(processor, max_seq=max_seq, mode=mode)
        companion = companion_processor.load_from_file(mode)
        self.context_by_imgid = {
            str(imgid): tokens
            for imgid, tokens in zip(companion['imgids'], companion['tokens'])
        }
        self.context_max_tokens = max(0, int(context_max_tokens))

    def _append_subtokens(self, bert_tokens, label_ids, eval_mask, token_type_ids, tokens, labels, segment_id, evaluate):
        for token, label in zip(tokens, labels):
            sub_tokens = self.tokenizer.tokenize(token) or [self.tokenizer.unk_token]
            for j, sub_tok in enumerate(sub_tokens):
                if len(bert_tokens) >= self.max_seq - 1:
                    return
                bert_tokens.append(sub_tok)
                expanded = self._expand_label_for_subword(label, j)
                label_ids.append(self.label_map.get(expanded, self.default_o_id))
                eval_mask.append(1 if evaluate and j == 0 else 0)
                token_type_ids.append(segment_id)

    def __getitem__(self, idx):
        tokens = self.data_dict['tokens'][idx]
        labels = self.data_dict['labels'][idx]
        imgid = str(self.data_dict['imgids'][idx])
        context_tokens = self.context_by_imgid.get(imgid, [])[: self.context_max_tokens]

        bert_tokens = [self.tokenizer.cls_token]
        label_ids = [self.default_o_id]
        eval_mask = [0]
        token_type_ids = [0]

        context_reserved = 0
        if context_tokens:
            context_reserved = min(self.context_max_tokens, max(0, self.max_seq // 2 - 2))
        main_max = self.max_seq - context_reserved - 2
        for token, label in zip(tokens, labels):
            sub_tokens = self.tokenizer.tokenize(token) or [self.tokenizer.unk_token]
            for j, sub_tok in enumerate(sub_tokens):
                if len(bert_tokens) >= main_max:
                    break
                bert_tokens.append(sub_tok)
                expanded = self._expand_label_for_subword(label, j)
                label_ids.append(self.label_map.get(expanded, self.default_o_id))
                eval_mask.append(1 if j == 0 else 0)
                token_type_ids.append(0)
            if len(bert_tokens) >= main_max:
                break

        bert_tokens.append(self.tokenizer.sep_token)
        label_ids.append(self.default_o_id)
        eval_mask.append(0)
        token_type_ids.append(0)

        if context_tokens and len(bert_tokens) < self.max_seq - 1:
            context_labels = ['O'] * len(context_tokens)
            self._append_subtokens(
                bert_tokens,
                label_ids,
                eval_mask,
                token_type_ids,
                context_tokens,
                context_labels,
                segment_id=1,
                evaluate=False,
            )
            if len(bert_tokens) < self.max_seq:
                bert_tokens.append(self.tokenizer.sep_token)
                label_ids.append(self.default_o_id)
                eval_mask.append(0)
                token_type_ids.append(1)

        input_ids = self.tokenizer.convert_tokens_to_ids(bert_tokens)
        attention_mask = [1] * len(input_ids)

        pad_len = self.max_seq - len(input_ids)
        if pad_len > 0:
            input_ids += [self.tokenizer.pad_token_id] * pad_len
            token_type_ids += [0] * pad_len
            attention_mask += [0] * pad_len
            label_ids += [self.default_o_id] * pad_len
            eval_mask += [0] * pad_len

        image = torch.zeros((3, 224, 224), dtype=torch.float)
        aux_imgs = torch.zeros((3, 3, self.aux_size, self.aux_size), dtype=torch.float)
        rcnn_imgs = torch.zeros((3, 3, self.rcnn_size, self.rcnn_size), dtype=torch.float)
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(token_type_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            torch.tensor(label_ids, dtype=torch.long),
            torch.tensor(eval_mask, dtype=torch.long),
            image,
            aux_imgs,
            rcnn_imgs,
        )


def collapse_label(label: str) -> str:
    if label == 'O':
        return 'O'
    if '-' in label:
        return label.split('-', 1)[1]
    return label


def collect_sequences(pred_ids, labels, eval_mask, attention_mask, id2label):
    y_true, y_pred = [], []
    for i in range(labels.size(0)):
        true_ids = labels[i].detach().cpu().tolist()
        eval_bits = eval_mask[i].detach().cpu().tolist()
        pred_full = pred_ids[i]
        sent_true, sent_pred = [], []
        for j, keep in enumerate(eval_bits):
            if keep != 1:
                continue
            sent_true.append(collapse_label(id2label.get(true_ids[j], 'O')))
            sent_pred.append(collapse_label(id2label.get(pred_full[j], 'O')))
        if sent_true:
            y_true.append(sent_true)
            y_pred.append(sent_pred)
    return y_true, y_pred


def collect_prediction_records(pred_ids, labels, eval_mask, id2label, dataset, start_index):
    records = []
    imgids = dataset.data_dict.get('imgids', [])
    tokens_by_sample = dataset.data_dict.get('tokens', [])
    for i in range(labels.size(0)):
        sample_index = start_index + i
        true_ids = labels[i].detach().cpu().tolist()
        eval_bits = eval_mask[i].detach().cpu().tolist()
        pred_full = pred_ids[i]
        gold, pred = [], []
        for j, keep in enumerate(eval_bits):
            if keep != 1:
                continue
            gold.append(collapse_label(id2label.get(true_ids[j], 'O')))
            pred.append(collapse_label(id2label.get(pred_full[j], 'O')))
        records.append(
            {
                'split_index': sample_index,
                'imgid': str(imgids[sample_index]) if sample_index < len(imgids) else str(sample_index),
                'tokens': tokens_by_sample[sample_index] if sample_index < len(tokens_by_sample) else [],
                'gold_labels': gold,
                'pred_labels': pred,
            }
        )
    return records


def compute_metrics(y_true, y_pred):
    flat_true = [x for sent in y_true for x in sent]
    flat_pred = [x for sent in y_pred for x in sent]
    eval_labels = sorted({lab for lab in set(flat_true + flat_pred) if lab != 'O'})
    if not eval_labels:
        return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'report': 'No non-O labels found.'}
    precision, recall, f1, _ = precision_recall_fscore_support(
        flat_true, flat_pred, labels=eval_labels, average='micro', zero_division=0
    )
    report = sk_classification_report(flat_true, flat_pred, labels=eval_labels, digits=4, zero_division=0)
    return {
        'f1': 100.0 * f1,
        'precision': 100.0 * precision,
        'recall': 100.0 * recall,
        'report': report,
    }


def _spans(seq):
    spans = []
    start = None
    label = None
    for idx, tag in enumerate(seq + ['O']):
        if tag == 'O':
            if label is not None:
                spans.append((label, start, idx - 1))
                start = None
                label = None
            continue
        if label is None:
            start = idx
            label = tag
        elif tag != label:
            spans.append((label, start, idx - 1))
            start = idx
            label = tag
    return spans


def compute_detailed_metrics(y_true, y_pred):
    flat_true = [x for sent in y_true for x in sent]
    flat_pred = [x for sent in y_pred for x in sent]
    eval_labels = sorted({lab for lab in set(flat_true + flat_pred) if lab != 'O'})
    token_per_class = {}
    if eval_labels:
        p, r, f1, support = precision_recall_fscore_support(
            flat_true,
            flat_pred,
            labels=eval_labels,
            average=None,
            zero_division=0,
        )
        token_per_class = {
            label: {
                'precision': 100.0 * float(pi),
                'recall': 100.0 * float(ri),
                'f1': 100.0 * float(fi),
                'support': int(si),
            }
            for label, pi, ri, fi, si in zip(eval_labels, p, r, f1, support)
        }

    true_spans = []
    pred_spans = []
    for sent_id, (truth, pred) in enumerate(zip(y_true, y_pred)):
        true_spans.extend((sent_id, *span) for span in _spans(truth))
        pred_spans.extend((sent_id, *span) for span in _spans(pred))
    true_set = set(true_spans)
    pred_set = set(pred_spans)
    tp = len(true_set & pred_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(true_set) if true_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        'token_per_class': token_per_class,
        'span': {
            'precision': 100.0 * precision,
            'recall': 100.0 * recall,
            'f1': 100.0 * f1,
            'support': len(true_set),
        },
    }


def build_pred_ids_argmax(logits, decode_bias=None):
    if decode_bias is not None:
        bias = torch.as_tensor(decode_bias, dtype=logits.dtype, device=logits.device)
        logits = logits + bias.view(1, 1, -1)
    return logits.argmax(dim=-1).detach().cpu().tolist()


def build_pred_ids_crf(model, emissions, attention_mask):
    paths = model.decode(emissions, attention_mask.bool())
    pred_ids = []
    attn_batch = attention_mask.detach().cpu().tolist()
    for i, attn in enumerate(attn_batch):
        seq = paths[i] if i < len(paths) else []
        full = [0] * len(attn)
        ptr = 0
        for j, m in enumerate(attn):
            if m == 1 and ptr < len(seq):
                full[j] = seq[ptr]
                ptr += 1
        pred_ids.append(full)
    return pred_ids


def token_ce_loss(logits, labels, eval_mask=None, class_weights=None):
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    if eval_mask is None:
        return criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
    active = eval_mask.view(-1).bool()
    if active.sum().item() == 0:
        return logits.sum() * 0.0
    return criterion(logits.view(-1, logits.size(-1))[active], labels.view(-1)[active])


def build_class_weights(num_labels, o_label_id, o_loss_weight, device):
    if abs(o_loss_weight - 1.0) < 1e-9:
        return None
    weights = torch.ones(num_labels, dtype=torch.float, device=device)
    weights[o_label_id] = float(o_loss_weight)
    return weights


def build_boundary_targets(labels, eval_mask, o_label_id):
    start_targets = torch.full_like(labels, -100)
    end_targets = torch.full_like(labels, -100)
    labels_cpu = labels.detach().cpu()
    eval_cpu = eval_mask.detach().cpu()
    for i in range(labels_cpu.size(0)):
        positions = [j for j, keep in enumerate(eval_cpu[i].tolist()) if keep == 1]
        seq = [int(labels_cpu[i, j].item()) for j in positions]
        for local_idx, (pos, lab) in enumerate(zip(positions, seq)):
            if lab == o_label_id:
                start_targets[i, pos] = 0
                end_targets[i, pos] = 0
                continue
            prev_lab = seq[local_idx - 1] if local_idx > 0 else o_label_id
            next_lab = seq[local_idx + 1] if local_idx + 1 < len(seq) else o_label_id
            start_targets[i, pos] = 1 if prev_lab != lab else 0
            end_targets[i, pos] = 1 if next_lab != lab else 0
    return start_targets.to(labels.device), end_targets.to(labels.device)


def boundary_loss(outputs, labels, eval_mask, o_label_id):
    start_targets, end_targets = build_boundary_targets(labels, eval_mask, o_label_id)
    start_loss = nn.CrossEntropyLoss(ignore_index=-100)(
        outputs['start_logits'].view(-1, 2),
        start_targets.view(-1),
    )
    end_loss = nn.CrossEntropyLoss(ignore_index=-100)(
        outputs['end_logits'].view(-1, 2),
        end_targets.view(-1),
    )
    return 0.5 * (start_loss + end_loss)


def evaluate(
    model,
    loader,
    device,
    id2label,
    model_type,
    o_label_id=0,
    boundary_loss_weight=0.0,
    loss_on_eval_tokens_only=False,
    decode_bias=None,
    class_weights=None,
    return_predictions=False,
):
    model.eval()
    all_true, all_pred = [], []
    prediction_records = []
    total_loss = 0.0
    steps = 0
    sample_offset = 0
    with torch.no_grad():
        for batch in loader:
            batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)
            input_ids, token_type_ids, attention_mask, labels, eval_mask, *_ = batch
            if model_type == 'bert_linear':
                logits = model(input_ids, attention_mask, token_type_ids)
                loss = token_ce_loss(
                    logits,
                    labels,
                    eval_mask if loss_on_eval_tokens_only else None,
                    class_weights=class_weights,
                )
                pred_ids = build_pred_ids_argmax(logits, decode_bias=decode_bias)
            elif model_type == 'bert_boundary':
                outputs = model(input_ids, attention_mask, token_type_ids)
                logits = outputs['logits']
                loss = token_ce_loss(logits, labels, eval_mask)
                if boundary_loss_weight > 0:
                    loss = loss + boundary_loss_weight * boundary_loss(outputs, labels, eval_mask, o_label_id)
                pred_ids = build_pred_ids_argmax(logits, decode_bias=decode_bias)
            else:
                emissions = model.emissions(input_ids, attention_mask, token_type_ids)
                loss = model.crf_loss(emissions, labels, attention_mask.bool())
                pred_ids = build_pred_ids_crf(model, emissions, attention_mask)
            total_loss += float(loss.detach().cpu().item())
            steps += 1
            y_true, y_pred = collect_sequences(pred_ids, labels, eval_mask, attention_mask, id2label)
            all_true.extend(y_true)
            all_pred.extend(y_pred)
            if return_predictions:
                prediction_records.extend(
                    collect_prediction_records(
                        pred_ids,
                        labels,
                        eval_mask,
                        id2label,
                        loader.dataset,
                        sample_offset,
                    )
                )
            sample_offset += labels.size(0)
    metrics = compute_metrics(all_true, all_pred)
    metrics['detailed'] = compute_detailed_metrics(all_true, all_pred)
    metrics['loss'] = total_loss / max(steps, 1)
    if return_predictions:
        metrics['predictions'] = prediction_records
    return metrics


def tune_decode_bias(model, loader, device, id2label, model_type, o_label_id, args):
    if model_type not in {'bert_linear', 'bert_boundary'} or not args.tune_decode_o_bias:
        return None, None
    num_labels = len(id2label)
    candidates = [round(x, 2) for x in np.arange(args.decode_o_bias_min, args.decode_o_bias_max + 1e-9, args.decode_o_bias_step)]
    best_bias = [0.0] * num_labels
    best_metrics = None
    for value in candidates:
        bias = [0.0] * num_labels
        bias[o_label_id] = float(value)
        metrics = evaluate(
            model,
            loader,
            device,
            id2label,
            model_type,
            o_label_id=o_label_id,
            boundary_loss_weight=args.boundary_loss_weight,
            loss_on_eval_tokens_only=args.loss_on_eval_tokens_only or model_type == 'bert_boundary',
            decode_bias=bias,
        )
        if best_metrics is None or metrics['f1'] > best_metrics['f1']:
            best_bias = bias
            best_metrics = metrics
    return best_bias, best_metrics


def train_one(args):
    device = resolve_device(args.device)
    set_seed(args.seed)

    log_dir = BASE_DIR / 'logs'
    report_dir = BASE_DIR / 'reports' / 'baseline_results'
    ckpt_dir = BASE_DIR / 'ckpt'
    log_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    exp_name = f"{args.model_type}_{args.dataset_name}_{time.strftime('%Y_%m_%d_%H_%M_%S')}"
    logger = logging.getLogger(exp_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_dir / exp_name)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s', '%m/%d/%Y %H:%M:%S'))
    logger.addHandler(fh)

    label_alias_map = None
    if args.dataset_name == 'caption_ner':
        label_alias_map = {
            'B-TENT': 'B-MENT',
            'I-TENT': 'I-MENT',
            'B-TOPI': 'B-MOPI',
            'I-TOPI': 'I-MOPI',
        }

    bert_source = resolve_cached_model_source(args.bert_name)
    processor = MNERProcessor(
        DATA_PATH[args.dataset_name],
        bert_source,
        collapse_bio=not args.use_bio_labels,
        include_test_in_label_map=False,
        label_alias_map=label_alias_map,
    )
    if args.paired_context != 'none':
        context_dataset_name = args.paired_context
        if context_dataset_name == 'auto':
            context_dataset_name = 'text_ner' if args.dataset_name == 'caption_ner' else 'caption_ner'
        context_label_alias_map = None
        if context_dataset_name == 'caption_ner':
            context_label_alias_map = {
                'B-TENT': 'B-MENT',
                'I-TENT': 'I-MENT',
                'B-TOPI': 'B-MOPI',
                'I-TOPI': 'I-MOPI',
            }
        companion_processor = MNERProcessor(
            DATA_PATH[context_dataset_name],
            bert_source,
            collapse_bio=not args.use_bio_labels,
            include_test_in_label_map=False,
            label_alias_map=context_label_alias_map,
        )
        train_dataset = PairedContextMNERDataset(
            processor,
            companion_processor,
            max_seq=args.max_seq,
            context_max_tokens=args.context_max_tokens,
            mode='train',
        )
        dev_dataset = PairedContextMNERDataset(
            processor,
            companion_processor,
            max_seq=args.max_seq,
            context_max_tokens=args.context_max_tokens,
            mode='dev',
        )
        test_dataset = PairedContextMNERDataset(
            processor,
            companion_processor,
            max_seq=args.max_seq,
            context_max_tokens=args.context_max_tokens,
            mode='test',
        )
        if not args.loss_on_eval_tokens_only and args.model_type in {'bert_linear', 'bert_boundary'}:
            logger.info('Paired context enabled: forcing loss_on_eval_tokens_only to avoid context O-label bias.')
            args.loss_on_eval_tokens_only = True
    else:
        train_dataset = MNERDataset(processor, max_seq=args.max_seq, mode='train')
        dev_dataset = MNERDataset(processor, max_seq=args.max_seq, mode='dev')
        test_dataset = MNERDataset(processor, max_seq=args.max_seq, mode='test')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    label_map = processor.get_label_map()
    id2label = {v: k for k, v in label_map.items()}
    num_labels = len(label_map)

    if args.model_type == 'bert_linear':
        model = BertLinearNER(bert_source, num_labels)
    elif args.model_type == 'bert_boundary':
        model = BertBoundaryNER(bert_source, num_labels)
    else:
        model = BertCRFNER(bert_source, num_labels)
    model.to(device)
    o_label_id = label_map.get('O', 0)
    class_weights = build_class_weights(num_labels, o_label_id, args.o_loss_weight, device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    train_steps = len(train_loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * train_steps),
        num_training_steps=train_steps,
    )

    best = {
        'dev_f1': -1.0,
        'epoch': None,
        'test_metrics': None,
        'dev_metrics': None,
        'state_dict': None,
    }

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)
            input_ids, token_type_ids, attention_mask, labels, eval_mask, *_ = batch
            optimizer.zero_grad()
            if args.model_type == 'bert_linear':
                logits = model(input_ids, attention_mask, token_type_ids)
                loss = token_ce_loss(
                    logits,
                    labels,
                    eval_mask if args.loss_on_eval_tokens_only else None,
                    class_weights=class_weights,
                )
            elif args.model_type == 'bert_boundary':
                outputs = model(input_ids, attention_mask, token_type_ids)
                logits = outputs['logits']
                loss = token_ce_loss(logits, labels, eval_mask)
                if args.boundary_loss_weight > 0:
                    loss = loss + args.boundary_loss_weight * boundary_loss(outputs, labels, eval_mask, o_label_id)
            else:
                emissions = model.emissions(input_ids, attention_mask, token_type_ids)
                loss = model.crf_loss(emissions, labels, attention_mask.bool())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += float(loss.detach().cpu().item())
            if step % args.log_steps == 0:
                print(f'[{exp_name}] epoch {epoch} step {step}/{len(train_loader)} loss={running / args.log_steps:.5f}')
                running = 0.0

        decode_bias = None
        tuned_dev_metrics = None
        if args.tune_decode_o_bias:
            decode_bias, tuned_dev_metrics = tune_decode_bias(
                model,
                dev_loader,
                device,
                id2label,
                args.model_type,
                o_label_id,
                args,
            )
        dev_metrics = tuned_dev_metrics or evaluate(
            model,
            dev_loader,
            device,
            id2label,
            args.model_type,
            o_label_id=o_label_id,
            boundary_loss_weight=args.boundary_loss_weight,
            loss_on_eval_tokens_only=args.loss_on_eval_tokens_only or args.model_type == 'bert_boundary',
            decode_bias=decode_bias,
            class_weights=class_weights,
        )
        test_metrics = evaluate(
            model,
            test_loader,
            device,
            id2label,
            args.model_type,
            o_label_id=o_label_id,
            boundary_loss_weight=args.boundary_loss_weight,
            loss_on_eval_tokens_only=args.loss_on_eval_tokens_only or args.model_type == 'bert_boundary',
            decode_bias=decode_bias,
            class_weights=class_weights,
        )
        test_metrics['decode_bias'] = decode_bias
        dev_metrics['decode_bias'] = decode_bias
        logger.info('Epoch %d/%d dev f1=%.4f precision=%.4f recall=%.4f', epoch, args.num_epochs, dev_metrics['f1'], dev_metrics['precision'], dev_metrics['recall'])
        logger.info('Epoch %d/%d test f1=%.4f precision=%.4f recall=%.4f', epoch, args.num_epochs, test_metrics['f1'], test_metrics['precision'], test_metrics['recall'])
        print(f'[{exp_name}] epoch {epoch}: dev_f1={dev_metrics["f1"]:.4f} test_f1={test_metrics["f1"]:.4f}')
        if dev_metrics['f1'] > best['dev_f1']:
            best = {
                'dev_f1': dev_metrics['f1'],
                'epoch': epoch,
                'test_metrics': test_metrics,
                'dev_metrics': dev_metrics,
                'decode_bias': decode_bias,
                'state_dict': copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()}),
            }
            torch.save(model.state_dict(), ckpt_dir / exp_name)

    prediction_path = None
    if args.export_predictions_dir and best.get('state_dict') is not None:
        model.load_state_dict(best['state_dict'])
        model.to(device)
        pred_metrics = evaluate(
            model,
            test_loader,
            device,
            id2label,
            args.model_type,
            o_label_id=o_label_id,
            boundary_loss_weight=args.boundary_loss_weight,
            loss_on_eval_tokens_only=args.loss_on_eval_tokens_only or args.model_type == 'bert_boundary',
            decode_bias=best.get('decode_bias'),
            class_weights=class_weights,
            return_predictions=True,
        )
        export_dir = Path(args.export_predictions_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        prediction_path = export_dir / f'{exp_name}_test_predictions.jsonl'
        with prediction_path.open('w', encoding='utf-8') as f:
            for record in pred_metrics.pop('predictions'):
                record.update(
                    {
                        'experiment_name': exp_name,
                        'dataset_name': args.dataset_name,
                        'model_type': args.model_type,
                        'bert_name': args.bert_name,
                        'seed': args.seed,
                        'best_epoch': best['epoch'],
                        'decode_bias': best.get('decode_bias'),
                        'num_epochs': args.num_epochs,
                        'batch_size': args.batch_size,
                        'lr': args.lr,
                        'warmup_ratio': args.warmup_ratio,
                        'max_seq': args.max_seq,
                        'paired_context': args.paired_context,
                        'context_max_tokens': args.context_max_tokens,
                    }
                )
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

    result = {
        'experiment_name': exp_name,
        'dataset_name': args.dataset_name,
        'model_type': args.model_type,
        'device': device,
        'best_epoch': best['epoch'],
        'best_dev_metrics': best['dev_metrics'],
        'final_test_metrics': best['test_metrics'],
        'final_test_detailed_metrics': best['test_metrics'].get('detailed', {}),
        'bert_name': args.bert_name,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'warmup_ratio': args.warmup_ratio,
        'max_seq': args.max_seq,
        'use_bio_labels': args.use_bio_labels,
        'boundary_loss_weight': args.boundary_loss_weight,
        'loss_on_eval_tokens_only': args.loss_on_eval_tokens_only,
        'o_loss_weight': args.o_loss_weight,
        'tune_decode_o_bias': args.tune_decode_o_bias,
        'best_decode_bias': best.get('decode_bias'),
        'paired_context': args.paired_context,
        'context_max_tokens': args.context_max_tokens,
        'prediction_path': str(prediction_path) if prediction_path else None,
    }
    out_path = report_dir / f'{exp_name}.json'
    out_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(out_path)
    return out_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', choices=['caption_ner', 'text_ner'], required=True)
    parser.add_argument('--model_type', choices=['bert_linear', 'bert_crf', 'bert_boundary'], required=True)
    parser.add_argument('--bert_name', default='bert-base-uncased')
    parser.add_argument('--num_epochs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--warmup_ratio', type=float, default=0.06)
    parser.add_argument('--max_seq', type=int, default=80)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--log_steps', type=int, default=10)
    parser.add_argument('--use_bio_labels', action='store_true')
    parser.add_argument('--boundary_loss_weight', type=float, default=0.2)
    parser.add_argument('--loss_on_eval_tokens_only', action='store_true')
    parser.add_argument('--o_loss_weight', type=float, default=1.0)
    parser.add_argument('--tune_decode_o_bias', action='store_true')
    parser.add_argument('--decode_o_bias_min', type=float, default=-1.5)
    parser.add_argument('--decode_o_bias_max', type=float, default=1.5)
    parser.add_argument('--decode_o_bias_step', type=float, default=0.25)
    parser.add_argument('--paired_context', choices=['none', 'auto', 'text_ner', 'caption_ner'], default='none')
    parser.add_argument('--context_max_tokens', type=int, default=48)
    parser.add_argument('--export_predictions_dir', default=None)
    return parser.parse_args()


if __name__ == '__main__':
    train_one(parse_args())
