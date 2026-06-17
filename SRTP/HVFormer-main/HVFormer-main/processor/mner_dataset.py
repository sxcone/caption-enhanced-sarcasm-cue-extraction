import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from PIL import Image

logger = logging.getLogger(__name__)


_IMAGE_INDEX_CACHE: Dict[str, str] = {}
_IMAGE_DIRS_CACHE: List[Path] = []


def _discover_image_dirs() -> List[Path]:
    global _IMAGE_DIRS_CACHE
    if _IMAGE_DIRS_CACHE:
        return _IMAGE_DIRS_CACHE

    project_data_root = Path(__file__).resolve().parents[3]
    image_dirs: List[Path] = []
    for candidate in sorted(project_data_root.glob("part*_img*")):
        nested = candidate / candidate.name
        target = nested if nested.is_dir() else candidate
        if target.is_dir():
            image_dirs.append(target)

    _IMAGE_DIRS_CACHE = image_dirs
    return _IMAGE_DIRS_CACHE


def _build_image_index() -> Dict[str, str]:
    global _IMAGE_INDEX_CACHE
    if _IMAGE_INDEX_CACHE:
        return _IMAGE_INDEX_CACHE

    image_index: Dict[str, str] = {}
    for image_dir in _discover_image_dirs():
        for image_path in image_dir.iterdir():
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            image_index.setdefault(image_path.stem, str(image_path))

    _IMAGE_INDEX_CACHE = image_index
    return _IMAGE_INDEX_CACHE


class MNERProcessor(object):
    def __init__(self, data_path: Dict[str, str], bert_name: str,
                 clip_processor=None, aux_processor=None, rcnn_processor=None,
                 collapse_bio: bool = False,
                 include_test_in_label_map: bool = False,
                 label_alias_map: Dict[str, str] = None):
        self.data_path = data_path
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(bert_name, use_fast=False, add_prefix_space=True)
        except TypeError:
            self.tokenizer = AutoTokenizer.from_pretrained(bert_name, use_fast=False)
        self.clip_processor = clip_processor
        self.aux_processor = aux_processor
        self.rcnn_processor = rcnn_processor
        self.collapse_bio = collapse_bio
        self.include_test_in_label_map = include_test_in_label_map
        self.label_alias_map = {"OS": "O", "N-MOPI": "I-MOPI"}
        if label_alias_map:
            self.label_alias_map.update(label_alias_map)

        self._all_samples = None
        self._label_list = None
        self._label_map = None

    def normalize_label(self, label: str) -> str:
        # Fix annotation typos/noise and optional dataset-specific aliases.
        label = self.label_alias_map.get(label, label)
        if self.collapse_bio and label != "O" and "-" in label:
            return label.split("-", 1)[1]
        return label

    def _read_samples(self, file_path: str) -> List[Dict]:
        samples = []
        cur_imgid = None
        cur_tokens = []
        cur_labels = []

        with open(file_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()

                if not line:
                    if cur_imgid is not None and cur_tokens:
                        samples.append({"imgid": cur_imgid, "tokens": cur_tokens, "labels": cur_labels})
                    cur_imgid = None
                    cur_tokens, cur_labels = [], []
                    continue

                if line.startswith("IMGID:"):
                    if cur_imgid is not None and cur_tokens:
                        samples.append({"imgid": cur_imgid, "tokens": cur_tokens, "labels": cur_labels})
                    cur_imgid = line.split(":", 1)[1].strip()
                    cur_tokens, cur_labels = [], []
                    continue

                parts = line.split("\t")
                if len(parts) != 2:
                    continue

                token, label = parts[0], self.normalize_label(parts[1])
                cur_tokens.append(token)
                cur_labels.append(label)

        if cur_imgid is not None and cur_tokens:
            samples.append({"imgid": cur_imgid, "tokens": cur_tokens, "labels": cur_labels})

        return samples

    def load_from_file(self, mode: str = "train") -> Dict:
        load_file = self.data_path[mode]
        logger.info("Loading NER data from %s", load_file)

        samples = self._read_samples(load_file)
        tokens = [s["tokens"] for s in samples]
        labels = [s["labels"] for s in samples]
        imgids = [s["imgid"] for s in samples]

        return {
            "tokens": tokens,
            "labels": labels,
            "imgids": imgids,
            "dataid": list(range(len(samples))),
        }

    def _build_label_mapping(self) -> Tuple[List[str], Dict[str, int]]:
        if self._label_list is not None and self._label_map is not None:
            return self._label_list, self._label_map

        labels = set()
        splits = ["train", "dev"]
        if self.include_test_in_label_map:
            splits.append("test")
        for split in splits:
            split_path = self.data_path.get(split)
            if not split_path:
                continue
            split_samples = self._read_samples(split_path)
            for s in split_samples:
                labels.update(s["labels"])

        labels = sorted(labels)
        if "O" in labels:
            labels.remove("O")
            labels = ["O"] + labels

        label_map = {label: idx for idx, label in enumerate(labels)}
        self._label_list = labels
        self._label_map = label_map
        return self._label_list, self._label_map

    def get_label_list(self) -> List[str]:
        return self._build_label_mapping()[0]

    def get_label_map(self) -> Dict[str, int]:
        return self._build_label_mapping()[1]


class MNERDataset(Dataset):
    def __init__(self, processor: MNERProcessor, max_seq: int = 128,
                 aux_size: int = 128, rcnn_size: int = 64,
                 mode: str = "train") -> None:
        self.processor = processor
        self.mode = mode
        self.max_seq = max_seq

        self.data_dict = self.processor.load_from_file(mode)
        self.tokenizer = self.processor.tokenizer

        self.label_map = self.processor.get_label_map()
        self.id2label = {v: k for k, v in self.label_map.items()}

        self.aux_size = aux_size
        self.rcnn_size = rcnn_size

        self.default_o_id = self.label_map.get("O", 0)
        self.image_index = _build_image_index()
        matched = sum(1 for imgid in self.data_dict["imgids"] if imgid in self.image_index)
        logger.info(
            "NER %s split matched %d/%d images across %d folders.",
            mode,
            matched,
            len(self.data_dict["imgids"]),
            len(_discover_image_dirs()),
        )

    def __len__(self):
        return len(self.data_dict["tokens"])

    def _expand_label_for_subword(self, label: str, sub_idx: int) -> str:
        if self.processor.collapse_bio:
            return label
        if sub_idx == 0:
            return label
        if label.startswith("B-"):
            return "I-" + label[2:]
        return label

    def __getitem__(self, idx):
        tokens = self.data_dict["tokens"][idx]
        labels = self.data_dict["labels"][idx]
        imgid = self.data_dict["imgids"][idx]

        bert_tokens = [self.tokenizer.cls_token]
        label_ids = [self.default_o_id]
        eval_mask = [0]

        for token, label in zip(tokens, labels):
            sub_tokens = self.tokenizer.tokenize(token)
            if not sub_tokens:
                sub_tokens = [self.tokenizer.unk_token]

            for j, sub_tok in enumerate(sub_tokens):
                if len(bert_tokens) >= self.max_seq - 1:
                    break
                bert_tokens.append(sub_tok)
                expanded = self._expand_label_for_subword(label, j)
                label_ids.append(self.label_map.get(expanded, self.default_o_id))
                eval_mask.append(1 if j == 0 else 0)

            if len(bert_tokens) >= self.max_seq - 1:
                break

        bert_tokens.append(self.tokenizer.sep_token)
        label_ids.append(self.default_o_id)
        eval_mask.append(0)

        input_ids = self.tokenizer.convert_tokens_to_ids(bert_tokens)
        token_type_ids = [0] * len(input_ids)
        attention_mask = [1] * len(input_ids)

        pad_len = self.max_seq - len(input_ids)
        if pad_len > 0:
            input_ids += [self.tokenizer.pad_token_id] * pad_len
            token_type_ids += [0] * pad_len
            attention_mask += [0] * pad_len
            label_ids += [self.default_o_id] * pad_len
            eval_mask += [0] * pad_len

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        label_ids = torch.tensor(label_ids, dtype=torch.long)
        eval_mask = torch.tensor(eval_mask, dtype=torch.long)

        image_path = self.image_index.get(str(imgid))
        if image_path is not None:
            image_obj = Image.open(image_path).convert("RGB")
        else:
            image_obj = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.processor.clip_processor is not None:
            image = self.processor.clip_processor(images=image_obj, return_tensors="pt")["pixel_values"].squeeze(0)
        else:
            image = torch.zeros((3, 224, 224), dtype=torch.float)
        aux_imgs = torch.zeros((3, 3, self.aux_size, self.aux_size), dtype=torch.float)
        rcnn_imgs = torch.zeros((3, 3, self.rcnn_size, self.rcnn_size), dtype=torch.float)

        return input_ids, token_type_ids, attention_mask, label_ids, eval_mask, image, aux_imgs, rcnn_imgs
