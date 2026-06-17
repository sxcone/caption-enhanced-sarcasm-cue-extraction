import torch
import torch.cuda
from torch import nn, Tensor, device
import torch.nn.functional as F
try:
    from torchcrf import CRF as _CRF
    _CRF_IS_TORCHCRF = True
except ImportError:
    # Some package mirrors publish TorchCRF with a capitalized module name.
    from TorchCRF import CRF as _CRF
    _CRF_IS_TORCHCRF = False

from .bert_vit_inter_base_model import BertVitInterBaseModel, get_extended_attention_mask, get_head_mask
from transformers.modeling_outputs import TokenClassifierOutput


class REClassifier(nn.Module):
    def __init__(self, re_label_mapping=None, config=None, tokenizer=None):
        super().__init__()
        self.text_config = config
        num_relation_labels = len(re_label_mapping)
        self.classifier = nn.Linear(2 * self.text_config.hidden_size, num_relation_labels)
        self.head_start = tokenizer.convert_tokens_to_ids("<s>")  # <s> id: 30522
        self.tail_start = tokenizer.convert_tokens_to_ids("<o>")  # <o> id: 30526

    def forward(self, input_ids=None, output_state=None):
        (output_state, vision_hidden_states, text_hidden_states) = output_state
        last_hidden_state, pooler_output = output_state.last_hidden_state, output_state.pooler_output
        bsz, seq_len, hidden_size = last_hidden_state.shape

        # Keep tensors on the same device as model outputs so MPS/CPU/CUDA all work.
        entity_hidden_state = last_hidden_state.new_zeros((bsz, 2 * hidden_size))
        for i in range(bsz):
            head_pos = input_ids[i].eq(self.head_start).nonzero(as_tuple=False)
            tail_pos = input_ids[i].eq(self.tail_start).nonzero(as_tuple=False)
            if head_pos.numel() == 0 or tail_pos.numel() == 0:
                continue
            head_idx = head_pos[0].item()
            tail_idx = tail_pos[0].item()
            head_hidden = last_hidden_state[i, head_idx, :].squeeze()
            tail_hidden = last_hidden_state[i, tail_idx, :].squeeze()
            entity_hidden_state[i] = torch.cat([head_hidden, tail_hidden], dim=-1)
        logits = self.classifier(entity_hidden_state)
        return logits


# Bert VIT
class BertVitInterReModel(nn.Module):
    def __init__(self,
                 re_label_mapping=None,
                 tokenizer=None,
                 args=None,
                 vision_config=None,
                 text_config=None,
                 clip_model_dict=None,
                 bert_model_dict=None, ):
        super().__init__()
        self.args = args
        self.vision_config = vision_config
        self.text_config = text_config

        vision_config.device = args.device
        self.model = BertVitInterBaseModel(vision_config, text_config, args)

        # test load:
        vision_names, text_names = [], []
        model_dict = self.model.state_dict()
        for name in model_dict:
            if 'vision' in name:
                clip_name = name.replace('vision_', '').replace('model.', '')
                if clip_name in clip_model_dict:
                    vision_names.append(clip_name)
                    model_dict[name] = clip_model_dict[clip_name]
            if 'text' in name:
                text_name = name.replace('text_', '').replace('model.', '')
                if text_name in bert_model_dict:
                    text_names.append(text_name)
                    model_dict[name] = bert_model_dict[text_name]
        self.model.load_state_dict(model_dict)
        self.model.resize_token_embeddings(len(tokenizer))
        self.args = args
        # RE classifier
        self.re_classifier = REClassifier(re_label_mapping=re_label_mapping, config=text_config,
                                          tokenizer=tokenizer)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
            images=None,
            aux_imgs=None,
            rcnn_imgs=None,
            task='re',
            epoch=0,
    ):
        output = self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,

                            pixel_values=images,
                            aux_values=aux_imgs,
                            rcnn_values=rcnn_imgs,
                            return_dict=True,
                            output_hidden_states=True, )
        if task == 're':
            bert_vit_logits = self.re_classifier(output_state=output, input_ids=input_ids)

            if labels is not None:
                label_ce_loss_fn = nn.CrossEntropyLoss()
                label_loss_bert_vit = label_ce_loss_fn(bert_vit_logits, labels.view(-1))

                return label_loss_bert_vit, bert_vit_logits


class BertViTInterNerModel(nn.Module):
    def __init__(self,
                 label_list,
                 args,
                 vision_config,
                 text_config,
                 clip_model_dict=None,
                 bert_model_dict=None,
                 logger=None, ):
        super(BertViTInterNerModel, self).__init__()
        self.args = args
        self.vision_config = vision_config
        self.text_config = text_config
        self.model = BertVitInterBaseModel(vision_config, text_config, self.args)

        if clip_model_dict is not None and bert_model_dict is not None:
            model_dict = self.model.state_dict()
            for name in model_dict:
                if 'vision' in name:
                    clip_name = name.replace('vision_', '').replace('model.', '')
                    if clip_name in clip_model_dict:
                        model_dict[name] = clip_model_dict[clip_name]
                if 'text' in name:
                    text_name = name.replace('text_', '').replace('model.', '')
                    if text_name in bert_model_dict:
                        model_dict[name] = bert_model_dict[text_name]
            self.model.load_state_dict(model_dict)

        self.num_labels = len(label_list)
        self.label_list = list(label_list)
        self.o_label_id = self.label_list.index("O") if "O" in self.label_list else 0
        if _CRF_IS_TORCHCRF:
            self.crf = _CRF(self.num_labels, batch_first=True)
            self._crf_style = "torchcrf"
        else:
            use_gpu = str(self.args.device).startswith("cuda")
            # TorchCRF's pad_idx path triggers an in-place autograd issue on recent PyTorch;
            # keep pad_idx unset and rely on the explicit mask during loss/decode.
            self.crf = _CRF(self.num_labels, pad_idx=None, use_gpu=use_gpu)
            self._crf_style = "TorchCRF"
        self.fc = nn.Linear(self.text_config.hidden_size, self.num_labels)
        self.use_span_boundary_head = bool(getattr(self.args, "use_span_boundary_head", False))
        self.boundary_loss_weight = float(getattr(self.args, "boundary_loss_weight", 0.2))
        if self.use_span_boundary_head:
            self.start_head = nn.Linear(self.text_config.hidden_size, 2)
            self.end_head = nn.Linear(self.text_config.hidden_size, 2)
        else:
            self.start_head = None
            self.end_head = None
        self.dropout = nn.Dropout(0.1)
        self.batch_id = 0
        # Using text-only hidden states for NER is more stable when images are unavailable.
        self.use_text_only_states = True
        self.ce_loss_weight = float(getattr(self.args, "ce_loss_weight", 1.0))
        self.crf_loss_weight = float(getattr(self.args, "crf_loss_weight", 1.0))
        self.decode_with_argmax = bool(getattr(self.args, "decode_with_argmax", False))
        self.ce_on_all_tokens = bool(getattr(self.args, "ce_on_all_tokens", False))
        self.focal_gamma = float(getattr(self.args, "focal_gamma", 0.0))
        self.fast_text_only = bool(getattr(self.args, "fast_text_only", True))
        self.text_layer_agg = str(getattr(self.args, "text_layer_agg", "last"))
        if self.text_layer_agg == "learned_last4":
            # Scalar mix over the top layers is a lightweight NER-oriented upgrade.
            self.layer_mix_logits = nn.Parameter(torch.zeros(4))
        else:
            self.register_parameter("layer_mix_logits", None)

        group_ids = []
        group_map = {}
        is_i_label = []
        for label in self.label_list:
            if label == "O":
                group_ids.append(-1)
                is_i_label.append(False)
                continue
            if "-" in label:
                prefix, group = label.split("-", 1)
            else:
                prefix, group = "B", label
            if group not in group_map:
                group_map[group] = len(group_map)
            group_ids.append(group_map[group])
            is_i_label.append(prefix == "I")
        self.register_buffer("label_group_ids", torch.tensor(group_ids, dtype=torch.long), persistent=False)
        self.register_buffer("label_is_i", torch.tensor(is_i_label, dtype=torch.bool), persistent=False)

    def _forward_text_only(self, input_ids, attention_mask, token_type_ids):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        input_shape = input_ids.size()
        device = input_ids.device
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device, dtype=torch.long)

        extended_attention_mask = get_extended_attention_mask(attention_mask, input_shape, device)
        head_mask = get_head_mask(None, self.model.text_config.num_hidden_layers)
        text_hidden_states = self.model.text_embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        )
        all_hidden_states = [text_hidden_states]
        for idx in range(self.model.text_config.num_hidden_layers):
            text_hidden_states = self.model.encoder.text_layer[idx](
                text_hidden_states,
                attention_mask=extended_attention_mask,
                head_mask=head_mask[idx],
                output_attentions=False,
                current_layer=idx,
            )[0]
            all_hidden_states.append(text_hidden_states)

        if self.text_layer_agg == "last":
            return text_hidden_states

        topk_states = all_hidden_states[-4:]
        if self.text_layer_agg == "mean_last4":
            return torch.stack(topk_states, dim=0).mean(dim=0)

        if self.text_layer_agg == "learned_last4":
            mix_weights = torch.softmax(self.layer_mix_logits, dim=0)
            mixed = 0.0
            for weight, hidden in zip(mix_weights, topk_states):
                mixed = mixed + weight * hidden
            return mixed

        return text_hidden_states

    def _crf_decode(self, emissions, mask):
        if hasattr(self.crf, "decode"):
            return self.crf.decode(emissions, mask)
        if hasattr(self.crf, "viterbi_decode"):
            return self.crf.viterbi_decode(emissions, mask)
        raise RuntimeError("Unsupported CRF implementation: no decode method found.")

    def _crf_loss(self, emissions, labels, mask):
        if self._crf_style == "torchcrf":
            return -1 * self.crf(emissions, labels, mask=mask, reduction='mean')

        # TorchCRF returns per-sample log-likelihood, so take mean manually.
        llh = self.crf(emissions, labels, mask)
        if isinstance(llh, torch.Tensor) and llh.dim() > 0:
            llh = llh.mean()
        return -1 * llh

    def _build_boundary_targets(self, labels, active_mask):
        start_targets = torch.zeros_like(labels, dtype=torch.long)
        end_targets = torch.zeros_like(labels, dtype=torch.long)

        labels_cpu = labels.detach().cpu()
        active_cpu = active_mask.detach().cpu()
        group_cpu = self.label_group_ids.detach().cpu()
        is_i_cpu = self.label_is_i.detach().cpu()

        for batch_idx in range(labels_cpu.size(0)):
            active_positions = active_cpu[batch_idx].nonzero(as_tuple=False).view(-1).tolist()
            for pos_idx, pos in enumerate(active_positions):
                cur_label = int(labels_cpu[batch_idx, pos].item())
                if cur_label == self.o_label_id:
                    continue

                cur_group = int(group_cpu[cur_label].item())
                prev_label = self.o_label_id
                next_label = self.o_label_id
                if pos_idx > 0:
                    prev_label = int(labels_cpu[batch_idx, active_positions[pos_idx - 1]].item())
                if pos_idx + 1 < len(active_positions):
                    next_label = int(labels_cpu[batch_idx, active_positions[pos_idx + 1]].item())

                prev_group = int(group_cpu[prev_label].item()) if prev_label != self.o_label_id else -1
                next_group = int(group_cpu[next_label].item()) if next_label != self.o_label_id else -1

                starts_span = (
                    not bool(is_i_cpu[cur_label].item())
                    or prev_label == self.o_label_id
                    or prev_group != cur_group
                )
                ends_span = next_label == self.o_label_id or next_group != cur_group
                if starts_span:
                    start_targets[batch_idx, pos] = 1
                if ends_span:
                    end_targets[batch_idx, pos] = 1

        return start_targets, end_targets

    def _boundary_loss(self, sequence_output, labels, active_mask):
        if not self.use_span_boundary_head or self.boundary_loss_weight <= 0:
            return sequence_output.new_tensor(0.0)
        if not active_mask.any():
            return sequence_output.new_tensor(0.0)

        start_targets, end_targets = self._build_boundary_targets(labels, active_mask)
        start_logits = self.start_head(sequence_output)
        end_logits = self.end_head(sequence_output)
        flat_active = active_mask.view(-1)
        start_loss = F.cross_entropy(
            start_logits.view(-1, 2)[flat_active],
            start_targets.view(-1)[flat_active],
            reduction="mean",
        )
        end_loss = F.cross_entropy(
            end_logits.view(-1, 2)[flat_active],
            end_targets.view(-1)[flat_active],
            reduction="mean",
        )
        return 0.5 * (start_loss + end_loss)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
            eval_mask=None,
            class_weights=None,
            images=None,
            aux_imgs=None,
            rcnn_imgs=None,
    ):
        if self.use_text_only_states and self.fast_text_only:
            sequence_output = self._forward_text_only(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        else:
            output = self.model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,

                                pixel_values=images,
                                aux_values=aux_imgs,
                                rcnn_values=rcnn_imgs,
                                return_dict=True,
                                output_hidden_states=True, )
            (output_state, _, text_hidden_states) = output
            if self.use_text_only_states:
                sequence_output = text_hidden_states  # bsz, len, hidden
            else:
                sequence_output = output_state.last_hidden_state  # bsz, len, hidden
        sequence_output = self.dropout(sequence_output)  # bsz, len, hidden
        emissions = self.fc(sequence_output)  # bsz, len, labels

        crf_mask = attention_mask.bool()
        loss = None
        if labels is not None:
            crf_loss = self._crf_loss(emissions, labels, crf_mask)

            # Auxiliary token-level CE helps avoid collapsing to all-O predictions.
            if eval_mask is None or self.ce_on_all_tokens:
                active_mask = crf_mask
            else:
                active_mask = eval_mask.bool() & crf_mask
            flat_logits = emissions.view(-1, self.num_labels)
            flat_labels = labels.view(-1)
            flat_active = active_mask.view(-1)
            if class_weights is not None:
                class_weights = class_weights.to(flat_logits.device)
            if flat_active.any():
                logits_active = flat_logits[flat_active]
                labels_active = flat_labels[flat_active]
                if self.focal_gamma > 0:
                    ce_per_token = F.cross_entropy(
                        logits_active,
                        labels_active,
                        weight=class_weights,
                        reduction="none",
                    )
                    probs = torch.softmax(logits_active, dim=-1)
                    true_prob = probs.gather(1, labels_active.unsqueeze(1)).squeeze(1)
                    true_prob = true_prob.clamp(min=1e-8, max=1.0)
                    focal_factor = torch.pow(1.0 - true_prob, self.focal_gamma)
                    token_ce = (focal_factor * ce_per_token).mean()
                else:
                    token_ce = F.cross_entropy(
                        logits_active,
                        labels_active,
                        weight=class_weights,
                        reduction="mean",
                    )
            else:
                token_ce = emissions.new_tensor(0.0)
            boundary_loss = self._boundary_loss(sequence_output, labels, active_mask)
            loss = (
                self.crf_loss_weight * crf_loss
                + self.ce_loss_weight * token_ce
                + self.boundary_loss_weight * boundary_loss
            )
        return TokenClassifierOutput(
            loss=loss,
            logits=emissions
        )
