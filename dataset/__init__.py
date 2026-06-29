"""
Dataset factory for VLAT / MVCM experiments.
"""
import json
import random

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset.randaugment import RandomAugment
from dataset.utils import pre_caption
from dataset.vqa_dataset_enhanced import VQADatasetEnhanced as vqa_dataset_enhanced


# ─────────────────────────────────────────────────────────────────────────────
# Pretrain dataset (single-view): returns (image, caption)
# ─────────────────────────────────────────────────────────────────────────────

class pretrain_dataset(Dataset):
    def __init__(self, ann_files, transform, max_words=40):
        self.ann = []
        for f in ann_files:
            self.ann += json.load(open(f, "r"))
        self.transform = transform
        self.max_words = max_words

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]
        caption = ann["caption"]
        if isinstance(caption, list):
            caption = random.choice(caption)
        caption = pre_caption(caption, self.max_words)
        image = Image.open(ann["image"]).convert("RGB")
        image = self.transform(image)
        return image, caption


# ─────────────────────────────────────────────────────────────────────────────
# Multi-view pretrain dataset: returns (image_v1, image_v2, caption)
# image_v1 and image_v2 are independently augmented views of the same image
# (SimCLR-style). For MIMIC data with true PA/Lateral pairs, image field
# should be a list [pa_path, lat_path]; otherwise we use augmentation.
# ─────────────────────────────────────────────────────────────────────────────

class pretrain_mv_dataset(Dataset):
    """
    Multi-view pretraining dataset for MVCM-FG.

    JSON format per record:
      {"image": "path/to/img.jpg", "caption": "..."}
      OR for true multi-view (MIMIC):
      {"image": ["path/pa.jpg", "path/lat.jpg"], "caption": "..."}

    Returns: (image_v1, image_v2, caption_str)
    """

    def __init__(self, ann_files, transform_v1, transform_v2, max_words=40):
        self.ann = []
        for f in ann_files:
            self.ann += json.load(open(f, "r"))
        self.transform_v1 = transform_v1
        self.transform_v2 = transform_v2
        self.max_words = max_words

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]
        caption = ann["caption"]
        if isinstance(caption, list):
            caption = random.choice(caption)
        caption = pre_caption(caption, self.max_words)

        img_field = ann["image"]
        if isinstance(img_field, list) and len(img_field) >= 2:
            # True multi-view (e.g. MIMIC PA + Lateral)
            img_v1 = Image.open(img_field[0]).convert("RGB")
            img_v2 = Image.open(img_field[1]).convert("RGB")
            image_v1 = self.transform_v1(img_v1)
            image_v2 = self.transform_v2(img_v2)
        else:
            # Simulated multi-view via two independent augmentations
            path = img_field[0] if isinstance(img_field, list) else img_field
            img = Image.open(path).convert("RGB")
            image_v1 = self.transform_v1(img)
            image_v2 = self.transform_v2(img)

        return image_v1, image_v2, caption


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

_NORMALIZE = transforms.Normalize(
    (0.48145466, 0.4578275, 0.40821073),
    (0.26862954, 0.26130258, 0.27577711),
)


def _make_pretrain_transform(image_res):
    return transforms.Compose([
        transforms.RandomResizedCrop(
            image_res, scale=(0.2, 1.0), interpolation=Image.BICUBIC
        ),
        transforms.RandomHorizontalFlip(),
        RandomAugment(
            2, 7, isPIL=True,
            augs=["Identity", "AutoContrast", "Equalize", "Brightness",
                  "Sharpness", "ShearX", "ShearY", "TranslateX",
                  "TranslateY", "Rotate"],
        ),
        transforms.ToTensor(),
        _NORMALIZE,
    ])


def _make_train_transform(image_res):
    return transforms.Compose([
        transforms.RandomResizedCrop(
            image_res, scale=(0.5, 1.0), interpolation=Image.BICUBIC
        ),
        transforms.RandomHorizontalFlip(),
        RandomAugment(
            2, 7, isPIL=True,
            augs=["Identity", "AutoContrast", "Equalize", "Brightness",
                  "Sharpness", "ShearX", "ShearY", "TranslateX",
                  "TranslateY", "Rotate"],
        ),
        transforms.ToTensor(),
        _NORMALIZE,
    ])


def _make_test_transform(image_res):
    return transforms.Compose([
        transforms.Resize((image_res, image_res), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        _NORMALIZE,
    ])


def create_dataset(dataset_type, config):
    image_res = config.get("image_res", 224)

    if dataset_type == "pretrain":
        t = _make_pretrain_transform(image_res)
        return pretrain_dataset(config["train_file"], t)

    if dataset_type == "pretrain_mv":
        t1 = _make_pretrain_transform(image_res)
        t2 = _make_pretrain_transform(image_res)
        return pretrain_mv_dataset(config["train_file"], t1, t2)

    if dataset_type in ("vqa", "vqa_enhanced"):
        train_t = _make_train_transform(image_res)
        test_t  = _make_test_transform(image_res)
        train_ds = vqa_dataset_enhanced(
            config["train_file"], train_t, split="train"
        )
        test_ds = vqa_dataset_enhanced(
            config["test_file"], test_t, split="test",
            answer_list=config.get("answer_list", ""),
        )
        return train_ds, test_ds

    raise ValueError(f"Unknown dataset type: {dataset_type!r}")


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    return [
        torch.utils.data.DistributedSampler(
            ds, num_replicas=num_tasks, rank=global_rank, shuffle=sh
        )
        for ds, sh in zip(datasets, shuffles)
    ]


def create_loader(datasets, samplers, batch_size, num_workers,
                  is_trains, collate_fns):
    loaders = []
    for ds, sampler, bs, nw, is_train, cfn in zip(
        datasets, samplers, batch_size, num_workers, is_trains, collate_fns
    ):
        shuffle  = is_train and sampler is None
        loader = DataLoader(
            ds,
            batch_size=bs,
            num_workers=nw,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=cfn,
            drop_last=is_train,
        )
        loaders.append(loader)
    return loaders


def vqa_collate_fn(batch):
    images, questions, answers, weights, ns = [], [], [], [], []
    for img, q, a, w in batch:
        images.append(img)
        questions.append(q)
        answers += a
        weights += w
        ns.append(len(a))
    return torch.stack(images, 0), questions, answers, torch.tensor(weights), ns
