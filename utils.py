import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

import config


def set_seed(seed=config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    return model


def false_negative_rate(y_true, y_pred, positive_class):
    """FNR for a given class treated as 'positive' (e.g. malignant)."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(config.NUM_CLASSES)))
    tp = cm[positive_class, positive_class]
    fn = cm[positive_class, :].sum() - tp
    return fn / (fn + tp) if (fn + tp) > 0 else float("nan")


def compute_metrics(y_true, y_pred, malignant_class=0):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "malignant_fnr": false_negative_rate(y_true, y_pred, malignant_class),
        "n": len(y_true),
    }
