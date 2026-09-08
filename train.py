import torch
import torch.nn as nn

import config
from data import get_dataloaders
from model import build_model
from utils import set_seed, save_checkpoint


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    with torch.set_grad_enabled(train):
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _ = get_dataloaders()
    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=config.LEARNING_RATE)

    best_val_loss = float("inf")
    for epoch in range(config.EPOCHS):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        print(f"Epoch {epoch + 1}/{config.EPOCHS} — train loss {train_loss:.4f}, val loss {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, config.CHECKPOINT_DIR / "baseline.pt")

    print(f"Best val loss: {best_val_loss:.4f}. Checkpoint saved to {config.CHECKPOINT_DIR / 'baseline.pt'}")


if __name__ == "__main__":
    main()
