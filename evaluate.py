import argparse
import pandas as pd
import torch

import config
from data import get_dataloaders
from model import build_model
from utils import load_checkpoint, compute_metrics


def collect_predictions(model, loader, device):
    model.eval()
    all_labels, all_preds, all_skin_types = [], [], []
    with torch.no_grad():
        for images, labels, skin_types in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu()
            all_labels.extend(labels.tolist())
            all_preds.extend(preds.tolist())
            all_skin_types.extend(skin_types.tolist())
    return all_labels, all_preds, all_skin_types


def main(checkpoint_name, output_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = get_dataloaders()

    model = build_model().to(device)
    model = load_checkpoint(model, config.CHECKPOINT_DIR / checkpoint_name, device)

    labels, preds, skin_types = collect_predictions(model, test_loader, device)
    df = pd.DataFrame({"label": labels, "pred": preds, "skin_type": skin_types})

    rows = []
    overall = compute_metrics(df["label"], df["pred"])
    overall["group"] = "overall"
    rows.append(overall)

    for group_name, types in config.SKIN_TONE_GROUPS.items():
        subset = df[df["skin_type"].isin(types)]
        if len(subset) == 0:
            continue
        metrics = compute_metrics(subset["label"], subset["pred"])
        metrics["group"] = group_name
        rows.append(metrics)

    for skin_type in sorted(df["skin_type"].unique()):
        subset = df[df["skin_type"] == skin_type]
        if len(subset) < 10:
            continue  # skip groups too small to report reliably
        metrics = compute_metrics(subset["label"], subset["pred"])
        metrics["group"] = f"type_{skin_type}"
        rows.append(metrics)

    results_df = pd.DataFrame(rows)[["group", "n", "accuracy", "recall_macro", "malignant_fnr"]]
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = config.RESULTS_DIR / output_name
    results_df.to_csv(output_path, index=False)
    print(results_df.to_string(index=False))
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="baseline.pt")
    parser.add_argument("--output", default="baseline_metrics.csv")
    args = parser.parse_args()
    main(args.checkpoint, args.output)
