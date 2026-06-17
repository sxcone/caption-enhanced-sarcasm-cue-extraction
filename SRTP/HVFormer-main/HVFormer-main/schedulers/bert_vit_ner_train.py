import json
import itertools
from pathlib import Path

import torch
from torch import optim
from tqdm import tqdm
from types import SimpleNamespace
from transformers.optimization import get_linear_schedule_with_warmup
from seqeval.metrics import classification_report as seq_classification_report
from seqeval.metrics import f1_score as seq_f1_score
from seqeval.metrics import precision_score as seq_precision_score
from seqeval.metrics import recall_score as seq_recall_score
from sklearn.metrics import classification_report as sk_classification_report
from sklearn.metrics import precision_recall_fscore_support


class BertVitNerTrainer(object):
    def __init__(self, train_data=None, dev_data=None, test_data=None,
                 label_map=None, id2label=None,
                 model=None, args=None, logger=None, writer=None) -> None:
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data

        self.label_map = label_map or {}
        self.id2label = id2label or {}

        self.model = model
        self.args = args
        self.logger = logger
        self.writer = writer

        self.best_dev_metrics = {"f1": 0.0, "precision": 0.0, "recall": 0.0}
        self.best_test_metrics = {"f1": 0.0, "precision": 0.0, "recall": 0.0}
        self.final_test_metrics = {"f1": 0.0, "precision": 0.0, "recall": 0.0}
        self.final_test_detailed_metrics = {}
        self.best_test_detailed_metrics = {}
        self.best_dev_epoch = None
        self.best_test_epoch = None
        self.decode_o_bias = 0.0
        self.best_decode_o_bias = 0.0
        self.decode_label_biases = {}
        self.best_decode_label_biases = {}

        if self.train_data is not None:
            self.train_num_steps = len(self.train_data) * args.num_epochs
        else:
            self.train_num_steps = 0

        self.step = 0
        self.refresh_step = args.log_steps
        self.optimizer = None
        self.scheduler = None
        self.class_weights = None

        self.before_train()

    def before_train(self):
        if getattr(self.args, "freeze_backbone", False):
            for name, param in self.model.named_parameters():
                if "fc" in name or "crf" in name or "start_head" in name or "end_head" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            self.logger.info("Freeze backbone enabled: only NER, CRF, and boundary heads are trainable.")

        optimizer_grouped_parameters = []
        base_params = {"lr": self.args.lr, "weight_decay": 1e-2, "params": []}
        crf_params = {"lr": self.args.crf_lr, "weight_decay": 0.0, "params": []}

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "crf" in name:
                crf_params["params"].append(param)
            else:
                base_params["params"].append(param)

        if base_params["params"]:
            optimizer_grouped_parameters.append(base_params)
        if crf_params["params"]:
            optimizer_grouped_parameters.append(crf_params)

        self.optimizer = optim.AdamW(optimizer_grouped_parameters, lr=self.args.lr)
        num_warmup_steps = int(self.args.warmup_ratio * self.train_num_steps)
        self.scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=self.train_num_steps,
        )
        self.model.to(self.args.device)
        self.logger.info("NER metric mode: %s", getattr(self.args, "ner_metric", "entity"))
        self.class_weights = self._build_class_weights()
        if self.class_weights is not None:
            o_id = self.label_map.get("O", 0)
            self.logger.info(
                "Use class weights (O=%s): %s",
                float(self.class_weights[o_id].item()) if o_id < len(self.class_weights) else "n/a",
                [round(float(v), 4) for v in self.class_weights.tolist()],
            )

    def _build_class_weights(self):
        if self.train_data is None or getattr(self.args, "disable_class_weights", False):
            return None
        num_labels = len(self.label_map)
        if num_labels <= 1:
            return None

        counts = torch.zeros(num_labels, dtype=torch.float)
        with torch.no_grad():
            for batch in self.train_data:
                attention_mask = batch[2]
                labels = batch[3]
                eval_mask = batch[4]
                active = (attention_mask == 1) & (eval_mask == 1)
                if not active.any():
                    continue
                active_labels = labels[active].view(-1)
                counts += torch.bincount(active_labels, minlength=num_labels).to(torch.float)

        if counts.sum() <= 0:
            return None

        power = float(getattr(self.args, "class_weight_power", 0.5))
        # Only reweight labels that actually appear in the training split.
        # Labels missing from train but present in dev/test should not distort scaling.
        present = counts > 0
        weights = torch.ones(num_labels, dtype=torch.float)
        if present.any():
            freq = counts[present] / counts[present].sum()
            present_weights = torch.pow(freq + 1e-12, -power)
            present_weights = present_weights / present_weights.mean()
            weights[present] = present_weights

        o_id = self.label_map.get("O", 0)
        if 0 <= o_id < len(weights):
            o_loss_weight = float(getattr(self.args, "o_loss_weight", 0.3))
            weights[o_id] = max(1e-4, o_loss_weight)

        max_cap = float(getattr(self.args, "class_weight_cap", 8.0))
        weights = torch.clamp(weights, min=1e-4, max=max_cap)
        return weights.to(self.args.device)

    def train(self):
        self.step = 0
        self.model.train()
        self.logger.info("***** Running NER training *****")
        self.logger.info("  Num instance = %d", len(self.train_data) * self.args.batch_size)
        self.logger.info("  Num epoch = %d", self.args.num_epochs)
        self.logger.info("  Batch size = %d", self.args.batch_size)
        self.logger.info("  Learning rate = %s", self.args.lr)

        if self.args.load_path is not None:
            self.logger.info("Loading model from %s", self.args.load_path)
            self.model.load_state_dict(torch.load(self.args.load_path, map_location="cpu"))
            self.logger.info("Load model successful")

        if self.args.do_test:
            self.logger.info("***** Start testing without training *****")
            self.test(0)
            return

        running_loss = 0.0
        for epoch in range(1, self.args.num_epochs + 1):
            print(f"\n===== Epoch {epoch}/{self.args.num_epochs} =====")
            for batch in self.train_data:
                self.step += 1
                batch = tuple(t.to(self.args.device) if isinstance(t, torch.Tensor) else t for t in batch)
                outputs, _, _ = self._step(batch)

                loss = outputs.loss
                running_loss += loss.detach().cpu().item()

                loss.backward()
                grad_clip_norm = float(getattr(self.args, "grad_clip_norm", 1.0))
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()

                if self.step % self.refresh_step == 0:
                    avg_loss = running_loss / self.refresh_step
                    print(f"Step {self.step}/{self.train_num_steps} - NER loss: {avg_loss:.5f}")
                    running_loss = 0.0

            if epoch >= self.args.eval_begin_epoch:
                self.evaluate(epoch)
                self.test(epoch)

            self.logger.info(
                "Best dev at epoch %s: f1=%.4f precision=%.4f recall=%.4f",
                self.best_dev_epoch,
                self.best_dev_metrics["f1"],
                self.best_dev_metrics["precision"],
                self.best_dev_metrics["recall"],
            )
            self.logger.info(
                "Best test at epoch %s: f1=%.4f precision=%.4f recall=%.4f",
                self.best_test_epoch,
                self.best_test_metrics["f1"],
                self.best_test_metrics["precision"],
                self.best_test_metrics["recall"],
            )
            self.logger.info(
                "Final test (selected by best dev epoch %s): f1=%.4f precision=%.4f recall=%.4f",
                self.best_dev_epoch,
                self.final_test_metrics["f1"],
                self.final_test_metrics["precision"],
                self.final_test_metrics["recall"],
            )
            self.logger.info(
                "Decode O-bias current=%.3f best_dev=%.3f",
                float(self.decode_o_bias),
                float(self.best_decode_o_bias),
            )
        self._write_summary()

    def _write_summary(self):
        report_dir = Path(self.args.save_path).resolve().parent.parent / "reports" / "baseline_results"
        report_dir.mkdir(parents=True, exist_ok=True)

        result = {
            "experiment_name": self.args.experiment_name,
            "dataset_name": self.args.dataset_name,
            "model_name": self.args.model_name,
            "device": self.args.device,
            "best_dev_epoch": self.best_dev_epoch,
            "best_test_epoch": self.best_test_epoch,
            "best_decode_o_bias": float(self.best_decode_o_bias),
            "best_dev_metrics": self.best_dev_metrics,
            "best_test_metrics": self.best_test_metrics,
            "final_test_metrics": self.final_test_metrics,
            "best_test_detailed_metrics": self.best_test_detailed_metrics,
            "final_test_detailed_metrics": self.final_test_detailed_metrics,
            "num_epochs": self.args.num_epochs,
            "batch_size": self.args.batch_size,
            "lr": self.args.lr,
            "warmup_ratio": self.args.warmup_ratio,
            "max_seq": self.args.max_seq,
            "text_layer_agg": getattr(self.args, "text_layer_agg", "last"),
            "fast_text_only": bool(getattr(self.args, "fast_text_only", True)),
            "disable_aux_images": bool(getattr(self.args, "disable_aux_images", False)),
            "disable_rcnn_regions": bool(getattr(self.args, "disable_rcnn_regions", False)),
            "disable_cross_modal": bool(getattr(self.args, "disable_cross_modal", False)),
            "disable_moe_fusion": bool(getattr(self.args, "disable_moe_fusion", False)),
            "decode_with_argmax": bool(getattr(self.args, "decode_with_argmax", False)),
            "tune_decode_o_bias": bool(getattr(self.args, "tune_decode_o_bias", False)),
            "tune_label_biases": bool(getattr(self.args, "tune_label_biases", False)),
            "best_decode_label_biases": {
                self.id2label.get(int(label_id), str(label_id)): float(bias)
                for label_id, bias in self.best_decode_label_biases.items()
            },
            "use_span_boundary_head": bool(getattr(self.args, "use_span_boundary_head", False)),
            "boundary_loss_weight": float(getattr(self.args, "boundary_loss_weight", 0.0)),
            "notes": getattr(self.args, "notes", ""),
        }
        out_path = report_dir / f"{self.args.experiment_name}.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        self.logger.info("Write experiment summary to %s", out_path)

    def _collect_sequences(self, outputs, labels, eval_mask, attention_mask, o_bias=0.0, label_biases=None):
        y_true, y_pred = [], []
        pred_source = outputs.logits

        pred_full_batch = None
        pred_paths = None
        if isinstance(pred_source, torch.Tensor):
            if getattr(self.args, "decode_with_argmax", False):
                logits = pred_source
                o_id = self.label_map.get("O", 0)
                if abs(float(o_bias)) > 1e-12 and 0 <= o_id < logits.size(-1):
                    logits = logits.clone()
                    logits[..., o_id] = logits[..., o_id] + float(o_bias)
                if label_biases:
                    logits = logits.clone()
                    for label_id, bias in label_biases.items():
                        if 0 <= int(label_id) < logits.size(-1) and abs(float(bias)) > 1e-12:
                            logits[..., int(label_id)] = logits[..., int(label_id)] + float(bias)
                pred_full_batch = logits.argmax(dim=-1).detach().cpu().tolist()
            else:
                pred_paths = self.model._crf_decode(pred_source, attention_mask.bool())
        else:
            pred_paths = pred_source

        for i in range(labels.size(0)):
            attn = attention_mask[i].detach().cpu().tolist()
            true_ids = labels[i].detach().cpu().tolist()
            eval_bits = eval_mask[i].detach().cpu().tolist()

            if pred_full_batch is not None:
                pred_full = pred_full_batch[i]
            else:
                pred_seq = pred_paths[i] if i < len(pred_paths) else []
                # decode outputs correspond to tokens where attention_mask == 1
                pred_full = [0] * len(attn)
                ptr = 0
                for j, m in enumerate(attn):
                    if m == 1 and ptr < len(pred_seq):
                        pred_full[j] = pred_seq[ptr]
                        ptr += 1

            sent_true, sent_pred = [], []
            for j, keep in enumerate(eval_bits):
                if keep != 1:
                    continue
                true_label = self.id2label.get(true_ids[j], "O")
                pred_label = self.id2label.get(pred_full[j], "O")
                sent_true.append(true_label)
                sent_pred.append(pred_label)

            if sent_true:
                y_true.append(sent_true)
                y_pred.append(sent_pred)

        return y_true, y_pred

    def _label_bias_candidates(self):
        if not getattr(self.args, "tune_label_biases", False):
            return [{}]
        o_id = self.label_map.get("O", 0)
        non_o_ids = sorted(int(label_id) for label, label_id in self.label_map.items() if label != "O")
        max_labels = int(getattr(self.args, "label_bias_max_labels", 4))
        if not non_o_ids or len(non_o_ids) > max_labels:
            self.logger.info(
                "Skip label-bias tuning: %d non-O labels exceeds max_labels=%d",
                len(non_o_ids),
                max_labels,
            )
            return [{}]

        bias_min = float(getattr(self.args, "label_bias_min", -0.5))
        bias_max = float(getattr(self.args, "label_bias_max", 0.5))
        bias_step = float(getattr(self.args, "label_bias_step", 0.25))
        if bias_step <= 0:
            bias_step = 0.25
        if bias_min > bias_max:
            bias_min, bias_max = bias_max, bias_min

        values = []
        num_steps = int((bias_max - bias_min) / bias_step) + 1
        num_steps = max(1, min(100, num_steps))
        for step_idx in range(num_steps):
            value = round(bias_min + step_idx * bias_step, 10)
            values.append(value)
        if 0.0 not in values:
            values.append(0.0)
            values = sorted(values)

        candidates = []
        for combo in itertools.product(values, repeat=len(non_o_ids)):
            candidate = {label_id: bias for label_id, bias in zip(non_o_ids, combo) if abs(float(bias)) > 1e-12}
            candidates.append(candidate)

        max_combinations = int(getattr(self.args, "label_bias_max_combinations", 2000))
        if len(candidates) > max_combinations:
            self.logger.info(
                "Trim label-bias candidates from %d to %d",
                len(candidates),
                max_combinations,
            )
            candidates = candidates[:max_combinations]
        self.logger.info(
            "Tune label biases for labels=%s with %d candidates",
            [self.id2label.get(label_id, str(label_id)) for label_id in non_o_ids if label_id != o_id],
            len(candidates),
        )
        return candidates

    def _compute_metrics(self, y_true, y_pred):
        if not y_true:
            return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "report": "No valid tokens."}
        metric_mode = getattr(self.args, "ner_metric", "entity")
        span_report = seq_classification_report(y_true, y_pred, digits=4)
        span_metrics = {
            "f1": 100.0 * seq_f1_score(y_true, y_pred),
            "precision": 100.0 * seq_precision_score(y_true, y_pred),
            "recall": 100.0 * seq_recall_score(y_true, y_pred),
            "report": span_report,
        }

        if metric_mode == "entity":
            return {
                "f1": span_metrics["f1"],
                "precision": span_metrics["precision"],
                "recall": span_metrics["recall"],
                "report": span_metrics["report"],
                "span": span_metrics,
                "token_per_class": {},
            }

        flatten_true = [lab for sent in y_true for lab in sent]
        flatten_pred = [lab for sent in y_pred for lab in sent]

        if metric_mode == "token_collapsed":
            def _collapse(label: str) -> str:
                if label == "O":
                    return "O"
                if "-" in label:
                    return label.split("-", 1)[1]
                return label

            flatten_true = [_collapse(lab) for lab in flatten_true]
            flatten_pred = [_collapse(lab) for lab in flatten_pred]

        eval_labels = sorted({lab for lab in set(flatten_true + flatten_pred) if lab != "O"})
        if not eval_labels:
            return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "report": "No non-O labels found."}

        precision, recall, f1, _ = precision_recall_fscore_support(
            flatten_true,
            flatten_pred,
            labels=eval_labels,
            average="micro",
            zero_division=0,
        )
        per_precision, per_recall, per_f1, per_support = precision_recall_fscore_support(
            flatten_true,
            flatten_pred,
            labels=eval_labels,
            average=None,
            zero_division=0,
        )
        token_per_class = {
            label: {
                "precision": 100.0 * float(p),
                "recall": 100.0 * float(r),
                "f1": 100.0 * float(class_f1),
                "support": int(support),
            }
            for label, p, r, class_f1, support in zip(
                eval_labels, per_precision, per_recall, per_f1, per_support
            )
        }
        report = sk_classification_report(
            flatten_true,
            flatten_pred,
            labels=eval_labels,
            digits=4,
            zero_division=0,
        )
        return {
            "f1": 100.0 * f1,
            "precision": 100.0 * precision,
            "recall": 100.0 * recall,
            "report": report,
            "span": span_metrics,
            "token_per_class": token_per_class,
        }

    def evaluate(self, epoch=0):
        self.model.eval()
        self.logger.info("***** Running NER evaluate *****")

        all_true, all_pred = [], []
        total_loss = 0.0
        cached_batches = []
        can_tune_decode_bias = bool(
            (
                getattr(self.args, "tune_decode_o_bias", False)
                or getattr(self.args, "tune_label_biases", False)
            )
            and getattr(self.args, "decode_with_argmax", False)
        )

        with torch.no_grad():
            with tqdm(total=len(self.dev_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Dev")
                for batch in self.dev_data:
                    batch = tuple(t.to(self.args.device) if isinstance(t, torch.Tensor) else t for t in batch)
                    outputs, labels, eval_mask, attention_mask = self._step(batch, return_attention=True)
                    total_loss += outputs.loss.detach().cpu().item()

                    if can_tune_decode_bias and isinstance(outputs.logits, torch.Tensor):
                        cached_batches.append(
                            (
                                outputs.logits.detach().cpu(),
                                labels.detach().cpu(),
                                eval_mask.detach().cpu(),
                                attention_mask.detach().cpu(),
                            )
                        )
                    else:
                        y_true, y_pred = self._collect_sequences(
                            outputs,
                            labels,
                            eval_mask,
                            attention_mask,
                            o_bias=self.decode_o_bias,
                            label_biases=self.decode_label_biases,
                        )
                        all_true.extend(y_true)
                        all_pred.extend(y_pred)
                    pbar.update()

        if cached_batches:
            bias_min = float(getattr(self.args, "decode_o_bias_min", -2.0))
            bias_max = float(getattr(self.args, "decode_o_bias_max", 2.0))
            bias_step = float(getattr(self.args, "decode_o_bias_step", 0.1))
            if bias_step <= 0:
                bias_step = 0.1
            if bias_min > bias_max:
                bias_min, bias_max = bias_max, bias_min

            o_bias_values = [float(self.decode_o_bias)]
            if getattr(self.args, "tune_decode_o_bias", False):
                o_bias_values = []
                num_steps = int((bias_max - bias_min) / bias_step) + 1
                num_steps = max(1, min(1000, num_steps))
                for step_idx in range(num_steps):
                    o_bias_values.append(bias_min + step_idx * bias_step)

            label_bias_values = self._label_bias_candidates()

            best_bias = self.decode_o_bias
            best_label_biases = dict(self.decode_label_biases)
            best_metrics = None
            best_true, best_pred = [], []

            for cand_bias in o_bias_values:
                for cand_label_biases in label_bias_values:
                    cand_true, cand_pred = [], []
                    for logits_cpu, labels_cpu, eval_cpu, attn_cpu in cached_batches:
                        out = SimpleNamespace(logits=logits_cpu)
                        y_true, y_pred = self._collect_sequences(
                            out,
                            labels_cpu,
                            eval_cpu,
                            attn_cpu,
                            o_bias=cand_bias,
                            label_biases=cand_label_biases,
                        )
                        cand_true.extend(y_true)
                        cand_pred.extend(y_pred)

                    cand_metrics = self._compute_metrics(cand_true, cand_pred)
                    better = (
                        best_metrics is None
                        or cand_metrics["f1"] > best_metrics["f1"] + 1e-12
                        or (
                            abs(cand_metrics["f1"] - best_metrics["f1"]) <= 1e-12
                            and cand_metrics["precision"] > best_metrics["precision"]
                        )
                    )
                    if better:
                        best_bias = cand_bias
                        best_label_biases = dict(cand_label_biases)
                        best_metrics = cand_metrics
                        best_true, best_pred = cand_true, cand_pred

            self.decode_o_bias = float(best_bias)
            self.decode_label_biases = dict(best_label_biases)
            all_true, all_pred = best_true, best_pred
            metrics = best_metrics
            self.logger.info(
                "Tune decode biases on dev: o_bias=%.3f label_biases=%s f1=%.4f precision=%.4f recall=%.4f",
                self.decode_o_bias,
                {
                    self.id2label.get(int(label_id), str(label_id)): round(float(bias), 4)
                    for label_id, bias in self.decode_label_biases.items()
                },
                metrics["f1"],
                metrics["precision"],
                metrics["recall"],
            )
        else:
            metrics = self._compute_metrics(all_true, all_pred)
        self.logger.info("\n%s", metrics["report"])
        self.logger.info(
            "Epoch %d/%d, dev f1=%.4f precision=%.4f recall=%.4f",
            epoch,
            self.args.num_epochs,
            metrics["f1"],
            metrics["precision"],
            metrics["recall"],
        )

        if metrics["f1"] >= self.best_dev_metrics["f1"]:
            self.best_dev_epoch = epoch
            self.best_dev_metrics.update({
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
            })
            self.best_decode_o_bias = float(self.decode_o_bias)
            self.best_decode_label_biases = dict(self.decode_label_biases)
            if self.args.save_path is not None and not getattr(self.args, "disable_model_save", False):
                torch.save(self.model.state_dict(), self.args.save_path)
                self.logger.info("Save best model at %s", self.args.save_path)

        self.model.train()

    def test(self, epoch=0):
        self.model.eval()
        self.logger.info("***** Running NER testing *****")

        if self.args.load_path is not None and self.args.do_test:
            self.logger.info("Loading model from %s", self.args.load_path)
            self.model.load_state_dict(torch.load(self.args.load_path, map_location="cpu"))

        all_true, all_pred = [], []
        with torch.no_grad():
            with tqdm(total=len(self.test_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Testing")
                for batch in self.test_data:
                    batch = tuple(t.to(self.args.device) if isinstance(t, torch.Tensor) else t for t in batch)
                    outputs, labels, eval_mask, attention_mask = self._step(batch, return_attention=True)

                    y_true, y_pred = self._collect_sequences(
                        outputs,
                        labels,
                        eval_mask,
                        attention_mask,
                        o_bias=self.decode_o_bias,
                        label_biases=self.decode_label_biases,
                    )
                    all_true.extend(y_true)
                    all_pred.extend(y_pred)
                    pbar.update()

        metrics = self._compute_metrics(all_true, all_pred)
        self.logger.info("\n%s", metrics["report"])
        self.logger.info(
            "Epoch %d/%d, test f1=%.4f precision=%.4f recall=%.4f",
            epoch,
            self.args.num_epochs,
            metrics["f1"],
            metrics["precision"],
            metrics["recall"],
        )

        if epoch == self.best_dev_epoch and metrics["f1"] >= self.final_test_metrics["f1"]:
            self.final_test_metrics.update({
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
            })
            self.final_test_detailed_metrics = {
                "span": metrics.get("span", {}),
                "token_per_class": metrics.get("token_per_class", {}),
            }

        if metrics["f1"] >= self.best_test_metrics["f1"]:
            self.best_test_epoch = epoch
            self.best_test_metrics.update({
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
            })
            self.best_test_detailed_metrics = {
                "span": metrics.get("span", {}),
                "token_per_class": metrics.get("token_per_class", {}),
            }

        self.model.train()

    def _step(self, batch, return_attention=False):
        input_ids, token_type_ids, attention_mask, labels, eval_mask, images, aux_imgs, rcnn_imgs = batch
        if getattr(self.args, "disable_aux_images", False):
            aux_imgs = None
        if getattr(self.args, "disable_rcnn_regions", False):
            rcnn_imgs = None
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            eval_mask=eval_mask,
            class_weights=self.class_weights,
            images=images,
            aux_imgs=aux_imgs,
            rcnn_imgs=rcnn_imgs,
        )

        if return_attention:
            return outputs, labels, eval_mask, attention_mask
        return outputs, labels, eval_mask
