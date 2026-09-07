# Fitzpatrick17k Skin Tone Bias Audit

Trains a 3-class skin condition classifier (malignant / benign / non-neoplastic)
on the Fitzpatrick17k dataset, then measures accuracy and false-negative rate
stratified by Fitzpatrick skin type (I-VI).

## Setup

```
pip install -r requirements.txt
```

Download the Fitzpatrick17k metadata CSV and images (see the dataset's GitHub repo
for the current image URLs/scraper) and place them under `data/`:

```
data/
  fitzpatrick17k.csv
  images/
    <image files>
```

## Run order

1. `python train.py` — trains the baseline classifier, saves a checkpoint to `checkpoints/`
2. `python evaluate.py` — loads the checkpoint, prints/saves overall + per-skin-type metrics
3. `python mitigate.py` — retrains with class reweighting, for comparison against step 2
4. Compare `results/baseline_metrics.csv` vs `results/mitigated_metrics.csv`

## Files

- `config.py` — paths, hyperparameters, class and skin-tone group mappings
- `data.py` — dataset class, transforms, dataloaders
- `model.py` — model definition (pretrained CNN, swapped classifier head)
- `train.py` — baseline training loop
- `evaluate.py` — stratified evaluation by Fitzpatrick type
- `mitigate.py` — class-reweighted training variant
- `utils.py` — metrics, seeding, checkpoint helpers
