import random
import os
import torch
import json
import ast
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from torchvision import transforms
import logging

logger = logging.getLogger(__name__)


class MREProcessor(object):
    def __init__(self, data_path, re_path, bert_name, clip_processor=None, aux_processor=None, rcnn_processor=None):
        self.data_path = data_path
        self.re_path = re_path
        self.tokenizer = BertTokenizer.from_pretrained(bert_name, do_lower_case=True)
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['<s>', '</s>', '<o>', '</o>']})
        self.clip_processor = clip_processor
        self.aux_processor = aux_processor
        self.rcnn_processor = rcnn_processor
        self._relation_dict = None

    def load_from_file(self, mode="train"):
        load_file = self.data_path[mode]
        logger.info("Loading data from {}".format(load_file))
        with open(load_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            words, relations, heads, tails, imgids, dataid = [], [], [], [], [], []
            for i, line in enumerate(lines):
                line = ast.literal_eval(line)  # str to dict
                words.append(line['token'])
                relations.append(line['relation'])
                heads.append(line['h'])  # {name, pos}
                tails.append(line['t'])
                imgids.append(line['img_id'])
                dataid.append(i)

        assert len(words) == len(relations) == len(heads) == len(tails) == (len(imgids))

        aux_imgs = {}
        rcnn_imgs = {}
        try:
            aux_path = self.data_path[mode + "_auximgs"]
            if os.path.exists(aux_path):
                aux_imgs = torch.load(aux_path)
        except Exception:
            logger.warning("No aux_imgs file found, using empty dict")
        
        try:
            rcnn_path = self.data_path[mode + '_img2crop']
            if os.path.exists(rcnn_path):
                rcnn_imgs = torch.load(rcnn_path)
        except Exception:
            logger.warning("No rcnn_imgs file found, using empty dict")
            
        return {'words': words, 'relations': relations, 'heads': heads, 'tails': tails, 'imgids': imgids,
                'dataid': dataid, 'aux_imgs': aux_imgs, "rcnn_imgs": rcnn_imgs}

    def get_relation_dict(self):
        if self._relation_dict is not None:
            return self._relation_dict

        re_dict = {}
        if os.path.exists(self.re_path):
            with open(self.re_path, 'r', encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    try:
                        re_dict = json.loads(content)
                    except json.JSONDecodeError:
                        # Support legacy files that are Python-literal dicts.
                        re_dict = ast.literal_eval(content)

        # Ensure all relations appearing in dataset splits are covered.
        all_relations = set()
        for split in ("train", "dev", "test"):
            split_path = self.data_path.get(split)
            if not split_path or not os.path.exists(split_path):
                continue
            with open(split_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = ast.literal_eval(line)
                    all_relations.add(item["relation"])

        if "None" in all_relations and "None" not in re_dict:
            re_dict["None"] = 0

        next_id = (max(re_dict.values()) + 1) if re_dict else 0
        for rel in sorted(all_relations):
            if rel not in re_dict:
                re_dict[rel] = next_id
                next_id += 1

        self._relation_dict = re_dict
        return self._relation_dict

    # relation and corresponding train samples
    def get_rel2id(self, train_path):
        re_dict = self.get_relation_dict()
        re2id = {key: [] for key in re_dict.keys()}
        with open(train_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = ast.literal_eval(line)  # str to dict
                assert line['relation'] in re2id
                re2id[line['relation']].append(i)
        return re2id


class MREDataset(Dataset):
    def __init__(self, processor, transform, img_path=None, aux_img_path=None, max_seq=40, aux_size=128, rcnn_size=64,
                 mode="train", write_path=None, do_test=False) -> None:
        self.processor = processor
        self.transform = transform
        self.max_seq = max_seq
        self.img_path = img_path[mode] if img_path is not None else img_path
        self.aux_img_path = aux_img_path[mode] if aux_img_path is not None else aux_img_path
        self.rcnn_img_path = 'data/mnre/'
        self.mode = mode
        self.data_dict = self.processor.load_from_file(mode)
        self.re_dict = self.processor.get_relation_dict()
        self.tokenizer = self.processor.tokenizer
        self.clip_processor = self.processor.clip_processor
        self.aux_processor = self.processor.aux_processor
        self.rcnn_processor = self.processor.rcnn_processor
        self.aux_size = aux_size
        self.rcnn_size = rcnn_size
        self.write_path = write_path
        self.do_test = do_test

    def __len__(self):
        return len(self.data_dict['words'])

    def __getitem__(self, idx):
        word_list, relation, head_d, tail_d, imgid = self.data_dict['words'][idx], self.data_dict['relations'][idx], \
                                                     self.data_dict['heads'][idx], self.data_dict['tails'][idx], \
                                                     self.data_dict['imgids'][idx]
        item_id = self.data_dict['dataid'][idx]
        # [CLS] ... <s> head </s> ... <o> tail <o/> .. [SEP]
        head_pos, tail_pos = head_d['pos'], tail_d['pos']
        # insert <s> <s/> <o> <o/>
        extend_word_list = []
        for i in range(len(word_list)):
            if i == head_pos[0]:
                extend_word_list.append('<s>')
            if i == head_pos[1]:
                extend_word_list.append('</s>')
            if i == tail_pos[0]:
                extend_word_list.append('<o>')
            if i == tail_pos[1]:
                extend_word_list.append('</o>')
            extend_word_list.append(word_list[i])
        extend_word_list = " ".join(extend_word_list)
        encode_dict = self.tokenizer(extend_word_list, max_length=self.max_seq, truncation=True,
                                     padding='max_length')
        input_ids, token_type_ids, attention_mask = encode_dict['input_ids'], encode_dict['token_type_ids'], \
                                                    encode_dict['attention_mask']
        input_ids, token_type_ids, attention_mask = torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(
            attention_mask)

        re_label = self.re_dict.get(relation, self.re_dict.get("None", 0))  # label to id

        # image process
        if self.img_path is not None:
            img_path = os.path.join(self.img_path, imgid)
            if os.path.exists(img_path):
                image = Image.open(img_path).convert('RGB')
            else:
                # Fallback to a blank image when a sample image is missing.
                image = Image.new('RGB', (224, 224), (0, 0, 0))
            image = self.clip_processor(images=image, return_tensors='pt')['pixel_values'].squeeze()
            if self.aux_img_path is not None:
                # detected object img
                aux_imgs = []
                aux_img_paths = []
                imgid = imgid.split(".")[0]
                if item_id in self.data_dict['aux_imgs']:
                    aux_img_paths = self.data_dict['aux_imgs'][item_id]
                    aux_img_paths = [os.path.join(self.aux_img_path, path) for path in aux_img_paths]

                # select 3 img
                for i in range(min(3, len(aux_img_paths))):
                    if not os.path.exists(aux_img_paths[i]):
                        continue
                    try:
                        aux_img = Image.open(aux_img_paths[i]).convert('RGB')
                        aux_img = self.aux_processor(images=aux_img, return_tensors='pt')['pixel_values'].squeeze()
                        aux_imgs.append(aux_img)
                    except Exception:
                        continue

                # padding
                for i in range(3 - len(aux_imgs)):
                    aux_imgs.append(torch.zeros((3, self.aux_size, self.aux_size)))

                aux_imgs = torch.stack(aux_imgs, dim=0)
                assert len(aux_imgs) == 3

                if self.rcnn_img_path is not None:
                    rcnn_imgs = []
                    rcnn_img_paths = []
                    if imgid in self.data_dict['rcnn_imgs']:
                        rcnn_img_paths = self.data_dict['rcnn_imgs'][imgid]
                        rcnn_img_paths = [os.path.join(self.rcnn_img_path, path) for path in rcnn_img_paths]

                    # select 3 img
                    for i in range(min(3, len(rcnn_img_paths))):
                        if not os.path.exists(rcnn_img_paths[i]):
                            continue
                        try:
                            rcnn_img = Image.open(rcnn_img_paths[i]).convert('RGB')
                            rcnn_img = self.rcnn_processor(images=rcnn_img, return_tensors='pt')['pixel_values'].squeeze()
                            rcnn_imgs.append(rcnn_img)
                        except Exception:
                            continue

                    # padding
                    for i in range(3 - len(rcnn_imgs)):
                        rcnn_imgs.append(torch.zeros((3, self.rcnn_size, self.rcnn_size)))

                    rcnn_imgs = torch.stack(rcnn_imgs, dim=0)
                    assert len(rcnn_imgs) == 3
                    if self.write_path is not None and self.mode == 'test' and self.do_test:
                        return input_ids, token_type_ids, attention_mask, torch.tensor(
                            re_label), image, aux_imgs, rcnn_imgs, extend_word_list, imgid
                    else:
                        return input_ids, token_type_ids, attention_mask, torch.tensor(
                            re_label), image, aux_imgs, rcnn_imgs

                return input_ids, token_type_ids, attention_mask, torch.tensor(re_label), image, aux_imgs

        return input_ids, token_type_ids, attention_mask, torch.tensor(re_label)
