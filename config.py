"""
Global configuration for the ASRS NLP pipeline.

This module centralises every tunable path, hyperparameter, label mapping and
model checkpoint so that the rest of the codebase stays declarative. Paths are
resolved relative to the repository root (parents[1] of this file).
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Directory layout (created on import and reference).
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
CHART_DIR = OUTPUT_DIR / "charts"
RESULT_DIR = OUTPUT_DIR / "results"
LOG_DIR = OUTPUT_DIR / "logs"

for d in (DATA_DIR, MODEL_DIR, OUTPUT_DIR, CHART_DIR, RESULT_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Raw cleaned-data files produced by prepare_dataset.py. Each file has the
# fixed header: acn,text,label,label_id  (label_id in 0..5).
TRAIN_CSV = DATA_DIR / "train.csv"
VAL_CSV = DATA_DIR / "val.csv"
TEST_CSV = DATA_DIR / "test.csv"

# --------------------------------------------------------------------------
# Primary-problem classification schema (6 classes from the cleaned data).
# --------------------------------------------------------------------------
LABELS = [
    "Human Factors",
    "Aircraft",
    "Ambiguous",
    "Procedure",
    "Company Policy",
    "Weather",
]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL = {idx: label for idx, label in enumerate(LABELS)}

# --------------------------------------------------------------------------
# Randomness.
# --------------------------------------------------------------------------
SEED = 42

# --------------------------------------------------------------------------
# Classifier hyper-parameters (mirrors the ASRS GitHub repo but for 6 labels).
# --------------------------------------------------------------------------
BASE_MODEL = "distilbert-base-uncased"   # classifier backbone
MAX_LEN = 256                            # tokeniser truncation length
LR = 2e-5
WEIGHT_DECAY = 0.01
PER_DEVICE_BATCH = 16
NUM_EPOCHS = 4
WARMUP_RATIO = 0.06

# By default the loss is weighted by the inverse class frequency so that the
# heavily imbalanced 6-class training set does not bias the model. Set to 0 in
# the CLI to turn weighting off.
CLASS_WEIGHTING = True

# --------------------------------------------------------------------------
# NER (token classification) configuration.
# --------------------------------------------------------------------------
# A pre-trained English token-classifier is used for the generic BIO tagging
# step (PER/ORG/LOC/MISC). The default is small enough to run on CPU and is
# automatically downloaded from the HuggingFace hub on first use.
NER_MODEL = "dslim/bert-base-NER"
# Set to "dslim/bert-large-NER" for higher quality but slower/larger inference.
NER_MAX_PIECES = 6     # max number of NER spans to keep per narrative
NER_MIN_GROUP = 2      # min tokens to treat a NER chain as one entity

# --------------------------------------------------------------------------
# Inference / evaluation.
# --------------------------------------------------------------------------
TOP_K = 3                        # number of class probabilities to show
EVAL_LIMIT = 0                   # 0 means "use the whole test set"
PREDICT_TABLE_LIMIT = 50         # rows for the demo prediction table when large