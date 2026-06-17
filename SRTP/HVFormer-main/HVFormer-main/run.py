import argparse
import logging
import os
import random
import sys
import time
import traceback
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import BertConfig, BertModel, CLIPConfig, CLIPProcessor

from models import BertVitInterReModel, BertViTInterNerModel
from processor import MREProcessor, MREDataset, MNERProcessor, MNERDataset
from schedulers import BertVitReTrainer, BertVitNerTrainer


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../../../"))
CAPTION_TEXT_DIR = os.path.join(PROJECT_ROOT, "caption和text")
CACHE_DIR = os.path.join(BASE_DIR, ".cache", "huggingface")

os.makedirs(CACHE_DIR, exist_ok=True)
os.environ.setdefault("HF_HOME", CACHE_DIR)
os.environ.setdefault("TRANSFORMERS_CACHE", CACHE_DIR)
# Force offline loading after local cache is prepared.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

warnings.filterwarnings("ignore", category=UserWarning)

DATA_PROCESS_CLASS = {
    "bert-vit-inter-re": (MREProcessor, MREDataset),
    "bert-vit-inter-ner": (MNERProcessor, MNERDataset),
}

MODEL_CLASS = {
    "bert-vit-inter-re": BertVitInterReModel,
    "bert-vit-inter-ner": BertViTInterNerModel,
}

DATA_PATH = {
    "mnre": {
        "train": os.path.join(BASE_DIR, "data/mnre/txt/ours_train.txt"),
        "dev": os.path.join(BASE_DIR, "data/mnre/txt/ours_val.txt"),
        "test": os.path.join(BASE_DIR, "data/mnre/txt/ours_test.txt"),
        "train_auximgs": os.path.join(BASE_DIR, "data/mnre/txt/mre_train_dict.pth"),
        "dev_auximgs": os.path.join(BASE_DIR, "data/mnre/txt/mre_dev_dict.pth"),
        "test_auximgs": os.path.join(BASE_DIR, "data/mnre/txt/mre_test_dict.pth"),
        "train_img2crop": os.path.join(BASE_DIR, "data/mnre/img_detect/train/train_img2crop.pth"),
        "dev_img2crop": os.path.join(BASE_DIR, "data/mnre/img_detect/val/val_img2crop.pth"),
        "test_img2crop": os.path.join(BASE_DIR, "data/mnre/img_detect/test/test_img2crop.pth"),
    },
    "text_ner": {
        "train": os.path.join(CAPTION_TEXT_DIR, "text_part1_1000_580.txt"),
        "dev": os.path.join(CAPTION_TEXT_DIR, "text_part2_650_364.txt"),
        "test": os.path.join(CAPTION_TEXT_DIR, "text_3545_part5_595_394.txt"),
    },
    "caption_ner": {
        "train": os.path.join(CAPTION_TEXT_DIR, "caption_3545_part1_1000_580.txt"),
        "dev": os.path.join(CAPTION_TEXT_DIR, "caption_part2_650_364.txt"),
        "test": os.path.join(CAPTION_TEXT_DIR, "caption_3545_part5_595_394.txt"),
    },
}

IMG_PATH = {
    "mnre": {
        "train": os.path.join(BASE_DIR, "data/mnre/img_org/train/"),
        "dev": os.path.join(BASE_DIR, "data/mnre/img_org/val/"),
        "test": os.path.join(BASE_DIR, "data/mnre/img_org/test/"),
    },
}

AUX_PATH = {
    "mnre": {
        "train": os.path.join(BASE_DIR, "data/mnre/img_vg/train/crops"),
        "dev": os.path.join(BASE_DIR, "data/mnre/img_vg/val/crops"),
        "test": os.path.join(BASE_DIR, "data/mnre/img_vg/test/crops"),
    },
}


def set_seed(seed=2022):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_logger(args):
    os.makedirs("logs", exist_ok=True)
    os.makedirs(args.save_path, exist_ok=True)
    if args.write_path is not None:
        os.makedirs(args.write_path, exist_ok=True)

    if args.do_test:
        exp_name = args.experiment_name
    else:
        exp_name = f"{args.experiment_name}_{args.dataset_name}_{time.strftime('%Y_%m_%d_%H_%M_%S')}"
    args.experiment_name = exp_name

    log_filename = os.path.join("logs", exp_name)
    args.save_path = os.path.join(args.save_path, exp_name)
    if args.write_path is not None:
        args.write_path = os.path.join(args.write_path, exp_name)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        filename=log_filename,
        level=logging.INFO,
    )
    return logging.getLogger(__name__)


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", default="test", type=str)
    parser.add_argument("--model_name", default="bert-vit-inter-re", type=str)
    parser.add_argument("--vit_name", default="openai/clip-vit-base-patch32", type=str)
    parser.add_argument("--dataset_name", default="mnre", type=str)
    parser.add_argument("--bert_name", default="bert-base-uncased", type=str)
    parser.add_argument("--num_epochs", default=20, type=int)
    parser.add_argument("--device", default="auto", type=str, help="auto/cuda/mps/cpu")
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--warmup_ratio", default=0.01, type=float)
    parser.add_argument("--eval_begin_epoch", default=1, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--load_path", default=None, type=str)
    parser.add_argument("--save_path", default="ckpt", type=str)
    parser.add_argument("--disable_model_save", action="store_true")
    parser.add_argument("--write_path", default="logs", type=str)
    parser.add_argument("--notes", default="", type=str)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument("--prompt_len", default=4, type=int)
    parser.add_argument("--max_seq", default=128, type=int)
    parser.add_argument("--aux_size", default=128, type=int)
    parser.add_argument("--rcnn_size", default=64, type=int)
    parser.add_argument("--log_mode", dest="log_mode", default="logger")
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--ignore_idx", default=0, type=int)
    parser.add_argument("--crf_lr", default=5e-2, type=float)
    parser.add_argument("--prompt_lr", default=3e-4, type=float)
    parser.add_argument("--log_steps", default=1, type=int)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--crf_loss_weight", default=1.0, type=float)
    parser.add_argument("--o_loss_weight", default=0.3, type=float)
    parser.add_argument("--class_weight_power", default=0.5, type=float)
    parser.add_argument("--class_weight_cap", default=8.0, type=float)
    parser.add_argument("--grad_clip_norm", default=1.0, type=float)
    parser.add_argument("--disable_class_weights", action="store_true")
    parser.add_argument("--decode_with_argmax", action="store_true")
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--ce_on_all_tokens", action="store_true")
    parser.add_argument("--focal_gamma", default=0.0, type=float)
    parser.add_argument("--collapse_bio_labels", action="store_true")
    parser.add_argument("--disable_fast_text_only", action="store_true")
    parser.add_argument(
        "--text_layer_agg",
        default="last",
        choices=["last", "mean_last4", "learned_last4"],
        type=str,
    )
    parser.add_argument("--disable_aux_images", action="store_true")
    parser.add_argument("--disable_rcnn_regions", action="store_true")
    parser.add_argument("--disable_cross_modal", action="store_true")
    parser.add_argument("--disable_moe_fusion", action="store_true")
    parser.add_argument(
        "--ner_metric",
        default="entity",
        choices=["entity", "token", "token_collapsed"],
        type=str,
    )
    parser.add_argument("--tune_decode_o_bias", action="store_true")
    parser.add_argument("--decode_o_bias_min", default=-2.0, type=float)
    parser.add_argument("--decode_o_bias_max", default=2.0, type=float)
    parser.add_argument("--decode_o_bias_step", default=0.1, type=float)
    parser.add_argument("--tune_label_biases", action="store_true")
    parser.add_argument("--label_bias_min", default=-0.5, type=float)
    parser.add_argument("--label_bias_max", default=0.5, type=float)
    parser.add_argument("--label_bias_step", default=0.25, type=float)
    parser.add_argument("--label_bias_max_labels", default=4, type=int)
    parser.add_argument("--label_bias_max_combinations", default=2000, type=int)
    parser.add_argument("--use_span_boundary_head", action="store_true")
    parser.add_argument("--boundary_loss_weight", default=0.2, type=float)
    args = parser.parse_args()
    args.fast_text_only = not args.disable_fast_text_only
    return args


def resolve_device(device_arg: str) -> str:
    if device_arg in ("cuda", "cpu"):
        return device_arg
    if device_arg == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                torch.ones(1, device="mps")
                return "mps"
            except RuntimeError as exc:
                print(f"Warning: requested MPS is unavailable at runtime ({exc}); falling back to CPU.")
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            torch.ones(1, device="mps")
            return "mps"
        except RuntimeError as exc:
            print(f"Warning: MPS is unavailable at runtime ({exc}); falling back to CPU.")
    return "cpu"


def resolve_cached_model_source(model_name_or_path: str) -> str:
    if os.path.exists(model_name_or_path):
        return model_name_or_path

    repo_name = model_name_or_path.replace("/", "--")
    snapshots_root = os.path.join(CACHE_DIR, "hub", f"models--{repo_name}", "snapshots")
    if not os.path.isdir(snapshots_root):
        return model_name_or_path

    usable = []
    for snap_name in os.listdir(snapshots_root):
        snap_dir = os.path.join(snapshots_root, snap_name)
        cfg = os.path.join(snap_dir, "config.json")
        pt = os.path.join(snap_dir, "pytorch_model.bin")
        sf = os.path.join(snap_dir, "model.safetensors")
        if os.path.isfile(cfg) and (os.path.isfile(pt) or os.path.isfile(sf)):
            usable.append(snap_dir)

    if usable:
        usable.sort()
        return usable[-1]
    return model_name_or_path


def _build_clip_processors(vit_source: str, args, require_processors=True):
    clip_processor, aux_processor, rcnn_processor = None, None, None
    if require_processors:
        try:
            clip_processor = CLIPProcessor.from_pretrained(vit_source)
            aux_processor = CLIPProcessor.from_pretrained(vit_source)
            aux_processor.image_processor.size = args.aux_size
            aux_processor.image_processor.crop_size = args.aux_size

            rcnn_processor = CLIPProcessor.from_pretrained(vit_source)
            rcnn_processor.image_processor.size = args.rcnn_size
            rcnn_processor.image_processor.crop_size = args.rcnn_size
        except OSError:
            class _ImageProcessorConfig:
                def __init__(self, size):
                    self.size = size
                    self.crop_size = size

            class _FallbackClipProcessor:
                def __init__(self, size):
                    self.image_processor = _ImageProcessorConfig(size)

                def __call__(self, images, return_tensors="pt"):
                    size = int(self.image_processor.size)
                    crop = int(self.image_processor.crop_size)
                    transform = transforms.Compose([
                        transforms.Resize(size, interpolation=InterpolationMode.BICUBIC),
                        transforms.CenterCrop(crop),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.48145466, 0.4578275, 0.40821073],
                            std=[0.26862954, 0.26130258, 0.27577711],
                        ),
                    ])
                    pixel_values = transform(images).unsqueeze(0)
                    return {"pixel_values": pixel_values}

            clip_processor = _FallbackClipProcessor(224)
            aux_processor = _FallbackClipProcessor(args.aux_size)
            rcnn_processor = _FallbackClipProcessor(args.rcnn_size)

    from transformers import CLIPVisionModel

    clip_vit = CLIPVisionModel.from_pretrained(vit_source)
    return clip_vit, clip_processor, aux_processor, rcnn_processor


def init_and_train_bert_vit_re(args, logger):
    data_process, dataset_class = DATA_PROCESS_CLASS[args.model_name]
    model_class = MODEL_CLASS[args.model_name]

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if args.do_train:
        re_data_path = DATA_PATH[args.dataset_name]
        img_path = IMG_PATH[args.dataset_name]
        aux_path = AUX_PATH[args.dataset_name]
        re_path = os.path.join(BASE_DIR, "data", args.dataset_name, "ours_rel2id.json")
        bert_source = resolve_cached_model_source(args.bert_name)
        vit_source = resolve_cached_model_source(args.vit_name)

        clip_vit, clip_processor, aux_processor, rcnn_processor = _build_clip_processors(
            vit_source, args, require_processors=True
        )

        processor = data_process(
            re_data_path,
            re_path,
            bert_source,
            clip_processor=clip_processor,
            aux_processor=aux_processor,
            rcnn_processor=rcnn_processor,
        )

        train_dataset = dataset_class(
            processor,
            transform,
            img_path,
            aux_path,
            args.max_seq,
            aux_size=args.aux_size,
            rcnn_size=args.rcnn_size,
            mode="train",
        )
        dev_dataset = dataset_class(
            processor,
            transform,
            img_path,
            aux_path,
            args.max_seq,
            aux_size=args.aux_size,
            rcnn_size=args.rcnn_size,
            mode="dev",
        )
        test_dataset = dataset_class(
            processor,
            transform,
            img_path,
            aux_path,
            args.max_seq,
            aux_size=args.aux_size,
            rcnn_size=args.rcnn_size,
            mode="test",
            write_path=args.write_path,
            do_test=args.do_test,
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        dev_dataloader = DataLoader(
            dev_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        re_dict = processor.get_relation_dict()
        tokenizer = processor.tokenizer

        vision_config = CLIPConfig.from_pretrained(vit_source).vision_config
        text_config = BertConfig.from_pretrained(bert_source)
        bert = BertModel.from_pretrained(bert_source)

        model = model_class(
            re_label_mapping=re_dict,
            tokenizer=tokenizer,
            args=args,
            vision_config=vision_config,
            text_config=text_config,
            clip_model_dict=clip_vit.state_dict(),
            bert_model_dict=bert.state_dict(),
        )

        trainer = BertVitReTrainer(
            train_data=train_dataloader,
            dev_data=dev_dataloader,
            test_data=test_dataloader,
            re_dict=re_dict,
            model=model,
            args=args,
            logger=logger,
            writer=None,
        )
        trainer.train()


def init_and_train_bert_vit_ner(args, logger):
    data_process, dataset_class = DATA_PROCESS_CLASS[args.model_name]
    model_class = MODEL_CLASS[args.model_name]

    if args.do_train:
        ner_data_path = DATA_PATH[args.dataset_name]
        bert_source = resolve_cached_model_source(args.bert_name)
        vit_source = resolve_cached_model_source(args.vit_name)

        clip_vit, clip_processor, aux_processor, rcnn_processor = _build_clip_processors(
            vit_source, args, require_processors=True
        )
        label_alias_map = None
        if args.dataset_name == "caption_ner":
            # Caption split contains a few text-domain label typos; map to caption schema.
            label_alias_map = {
                "B-TENT": "B-MENT",
                "I-TENT": "I-MENT",
                "B-TOPI": "B-MOPI",
                "I-TOPI": "I-MOPI",
            }

        processor = data_process(
            ner_data_path,
            bert_source,
            clip_processor=clip_processor,
            aux_processor=aux_processor,
            rcnn_processor=rcnn_processor,
            collapse_bio=args.collapse_bio_labels,
            include_test_in_label_map=False,
            label_alias_map=label_alias_map,
        )

        train_dataset = dataset_class(
            processor,
            max_seq=args.max_seq,
            aux_size=args.aux_size,
            rcnn_size=args.rcnn_size,
            mode="train",
        )
        dev_dataset = dataset_class(
            processor,
            max_seq=args.max_seq,
            aux_size=args.aux_size,
            rcnn_size=args.rcnn_size,
            mode="dev",
        )
        test_dataset = dataset_class(
            processor,
            max_seq=args.max_seq,
            aux_size=args.aux_size,
            rcnn_size=args.rcnn_size,
            mode="test",
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        dev_dataloader = DataLoader(
            dev_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        label_list = processor.get_label_list()
        label_map = processor.get_label_map()
        id2label = {v: k for k, v in label_map.items()}

        vision_config = CLIPConfig.from_pretrained(vit_source).vision_config
        text_config = BertConfig.from_pretrained(bert_source)
        bert = BertModel.from_pretrained(bert_source)

        model = model_class(
            label_list=label_list,
            args=args,
            vision_config=vision_config,
            text_config=text_config,
            clip_model_dict=clip_vit.state_dict(),
            bert_model_dict=bert.state_dict(),
        )

        trainer = BertVitNerTrainer(
            train_data=train_dataloader,
            dev_data=dev_dataloader,
            test_data=test_dataloader,
            label_map=label_map,
            id2label=id2label,
            model=model,
            args=args,
            logger=logger,
            writer=None,
        )
        trainer.train()


if __name__ == "__main__":
    try:
        args = parse_argument()
        args.device = resolve_device(args.device)
        set_seed(args.seed)
        logger = get_logger(args)
        logger.info(args)

        TRAINER = {
            "bert-vit-inter-re": init_and_train_bert_vit_re,
            "bert-vit-inter-ner": init_and_train_bert_vit_ner,
        }

        if args.model_name not in TRAINER:
            raise ValueError(f"The model {args.model_name} is not implemented!")

        TRAINER[args.model_name](args, logger)
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        try:
            logger.info(traceback.format_exc())
        except Exception:
            pass
        sys.exit(1)
    finally:
        print("run.py completed")
