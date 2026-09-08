from pathlib import Path

# Paths
DATA_DIR = Path("data")
CSV_URL = "https://raw.githubusercontent.com/mattgroh/fitzpatrick17k/main/fitzpatrick17k.csv"
IMAGE_DIR = DATA_DIR / "images"
CHECKPOINT_DIR = Path("checkpoints")
RESULTS_DIR = Path("results")

# Training
SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 1 # starting with 8 epochs, can increase if needed
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Labels
# Original Fitzpatrick17k paper collapses 114 fine-grained conditions into 3 super-classes.
# Updated this mapping after inspecting the CSV's label column names
SUPERCLASS_MAP = {
    "malignant": 0,
    "benign": 1,
    "non-neoplastic": 2,
}
NUM_CLASSES = len(SUPERCLASS_MAP)

# Skin tone grouping for stratified evaluation
# Fitzpatrick scale is 1-6
# group into light/dark for enough sample size per group,
SKIN_TONE_GROUPS = {
    "light": [1, 2, 3],
    "dark": [4, 5, 6],
}
