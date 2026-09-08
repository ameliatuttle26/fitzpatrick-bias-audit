import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

import config


class FitzpatrickDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = config.IMAGE_DIR / f"{row['md5hash']}.jpg"
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = config.SUPERCLASS_MAP[row["three_partition_label"]]
        skin_type = int(row["fitzpatrick_scale"])
        return image, label, skin_type


def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


def load_splits():
    df = pd.read_csv(config.CSV_URL)
    # Filter out rows with missing or invalid Fitzpatrick labels (some entries are unlabeled)
    df = df[df["fitzpatrick_scale"].between(1, 6)]
    df = df.head(300)

    train_df, temp_df = train_test_split(
        df, test_size=config.VAL_SPLIT + config.TEST_SPLIT,
        stratify=df["three_partition_label"], random_state=config.SEED,
    )
    relative_test_size = config.TEST_SPLIT / (config.VAL_SPLIT + config.TEST_SPLIT)
    val_df, test_df = train_test_split(
        temp_df, test_size=relative_test_size,
        stratify=temp_df["three_partition_label"], random_state=config.SEED,
    )
    return train_df, val_df, test_df


def get_dataloaders():
    train_df, val_df, test_df = load_splits()
    train_tf, eval_tf = get_transforms()

    train_loader = DataLoader(
        FitzpatrickDataset(train_df, train_tf),
        batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2,
    )
    val_loader = DataLoader(
        FitzpatrickDataset(val_df, eval_tf),
        batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2,
    )
    test_loader = DataLoader(
        FitzpatrickDataset(test_df, eval_tf),
        batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2,
    )
    return train_loader, val_loader, test_loader
