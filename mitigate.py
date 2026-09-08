import torch
import torch.nn as nn

import config
from data import get_dataloaders, load_splits
from model import build_model
from train import run_epoch
from utils import set_seed, save_checkpoint


def compute_class_weights(train_df, device):
    counts = train_df["three_partition_label"].map(config.SUPERCLASS_MAP).value_counts().sort_index()
    weights = 1.0 / counts
    weights = weights / weights.sum() * config.NUM_CLASSES
    return torch.tensor(weights.values, dtype=torch.float32, device=device)


def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _ = get_dataloaders()
    train_df, _, _ = load_splits()

    model = build_model().to(device)
    class_weights = compute_class_weights(train_df, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    best_val_loss = float("inf")
    for epoch in range(config.EPOCHS):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        print(f"Epoch {epoch + 1}/{config.EPOCHS} — train loss {train_loss:.4f}, val loss {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, config.CHECKPOINT_DIR / "mitigated.pt")

    print(f"Best val loss: {best_val_loss:.4f}. Checkpoint saved to {config.CHECKPOINT_DIR / 'mitigated.pt'}")
    print("Run: python evaluate.py --checkpoint mitigated.pt --output mitigated_metrics.csv")


if __name__ == "__main__":
    main()
