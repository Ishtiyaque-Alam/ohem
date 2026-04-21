"""
Datasets for APTOS 2019 and HAM10000.
"""
import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


# Standard ImageNet stats
MEAN = [0.485, 0.456, 0.406]
STD = [0.224, 0.224, 0.224]

IMG_SIZE = 224  # MobileNetV3-Large native resolution

CLASS_MAP_HAM = {
    "akiec": 0,
    "bcc": 1,
    "bkl": 2,
    "df": 3,
    "mel": 4,
    "nv": 5,
    "vasc": 6,
}


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

    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


class APTOSDataset(Dataset):
    """
    CSV columns expected:
        train CSV -> id_code, diagnosis
        test CSV  -> id_code (no label)
    """

    def __init__(self, csv_path: str, img_dir: str, split: str = "train"):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.tfm = get_transforms(split)
        self.has_labels = "diagnosis" in self.df.columns

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = os.path.join(self.img_dir, row["id_code"] + ".png")
        img = Image.open(path).convert("RGB")
        img = self.tfm(img)

        if self.has_labels:
            label = int(row["diagnosis"])
            return img, label
        return img, -1  # unknown label for test set


def find_image_path(img_id: str, dir1: str, dir2: str) -> str | None:
    for folder in (dir1, dir2):
        for ext in ("jpg", "jpeg", "png"):
            p = os.path.join(folder, f"{img_id}.{ext}")
            if os.path.exists(p):
                return p
    return None


def build_ham_dataframe(csv_path: str, img_dir_1: str, img_dir_2: str) -> pd.DataFrame:
    """
    Build HAM10000 dataframe with resolved file paths and mapped integer labels.
    """
    df = pd.read_csv(csv_path)
    df["label"] = df["dx"].map(CLASS_MAP_HAM)
    df["filepath"] = df["image_id"].apply(
        lambda x: find_image_path(x, img_dir_1, img_dir_2)
    )
    df = df.dropna(subset=["label", "filepath"]).copy()
    df["label"] = df["label"].astype(int)
    return df


class HAM10000Dataset(Dataset):
    """
    Expects a dataframe containing:
        - filepath : absolute/relative image path
        - label    : integer class id in [0, 6]
    """

    def __init__(self, df: pd.DataFrame, split: str = "train"):
        self.df = df.reset_index(drop=True)
        self.tfm = get_transforms(split)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["filepath"]).convert("RGB")
        img = self.tfm(img)
        return img, int(row["label"])
