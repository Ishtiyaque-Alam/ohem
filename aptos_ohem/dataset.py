"""
APTOS 2019 Blindness Detection – Dataset
"""
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


# ── standard ImageNet stats ──────────────────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.224, 0.224, 0.224]

IMG_SIZE = 224   # MobileNetV3-Large native resolution


def get_transforms(split: str) -> T.Compose:
    if split == "train":
        return T.Compose([
            T.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
            T.RandomCrop(IMG_SIZE),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])
    else:
        return T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])


class APTOSDataset(Dataset):
    """
    CSV columns expected:
        train CSV  →  id_code, diagnosis
        test  CSV  →  id_code              (no label)
    """

    def __init__(self, csv_path: str, img_dir: str, split: str = "train"):
        self.df      = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.split   = split
        self.tfm     = get_transforms(split)
        self.has_labels = "diagnosis" in self.df.columns

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row  = self.df.iloc[idx]
        path = os.path.join(self.img_dir, row["id_code"] + ".png")
        img  = Image.open(path).convert("RGB")
        img  = self.tfm(img)

        if self.has_labels:
            label = int(row["diagnosis"])
            return img, label
        return img, -1          # unknown label for test set
