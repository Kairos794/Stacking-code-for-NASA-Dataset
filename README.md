# Stacking and Extended Ensembles for NASA ASRS Six-Class Classification

Code repository for the second system reported in the MFE310TC coursework: a
Logistic Regression stacking model built on the prediction probabilities of
eight single-label Transformer classifiers and six per-class binary
classifiers, together with the extended probability-averaging ensembles used
for comparison.

The companion repository
[MFE310TC-Coursework-code](https://github.com/Kairos794/MFE310TC-Coursework-code)
contains the five-model probability-averaging system (the principal system
evaluated in the report) and the shared `asrs_lib` library. Both repositories
are provided because the two configurations reach comparable test performance.

## What this repository contains

```
Stacking-code-for-NASA-Dataset/
├── README.md
├── requirements.txt          ← pinned dependency versions (measured environment)
├── environment.md            ← software / hardware / configuration record
├── .gitignore
├── run_all.ps1               ← convenience script for the full pipeline
├── asrs_lib/
│   ├── config.py             ← paths, label mapping, hyper-parameters, seed
│   ├── data_utils.py         ← data loading and class-weight computation
│   └── classifier.py         ← training, inference (softmax_logits), metrics
└── scripts/
    ├── 01_prepare.py             ← validate the fixed split, ACN leakage check
    ├── 02_train_classifier.py    ← fine-tune one backbone
    ├── 03_ensemble.py            ← majority vote / probability averaging
    ├── 04_binary_per_class.py    ← six one-vs-rest binary classifiers
    ├── 05_hybrid.py              ← per-class hybrid of softmax and binary scores
    ├── 06_stack_ensemble.py      ← 84-feature stacking with Logistic Regression
    ├── 07_bias_tune_all.py       ← per-class log-bias tuning on the averaged pool
    ├── 08_metric_weights.py      ← alternative weighting schemes for evaluation
    └── 09_member_metrics.py      ← per-member metrics from cached probabilities
```

All scripts import the library through `asrs_lib`, which is resolved relative
to the repository root, so the commands below can be executed from the
repository root without any path edits.

## Data

The pipeline reads three fixed CSV files from `data/`:

```
data/train.csv   data/val.csv   data/test.csv
```

Each file carries the header `acn,text,label,label_id`, where `label_id` is an
integer in `0..5` mapped to:

| id | label |
|---:|---|
| 0 | Human Factors |
| 1 | Aircraft |
| 2 | Ambiguous |
| 3 | Procedure |
| 4 | Company Policy |
| 5 | Weather |

Split sizes used in the experiments: 69,501 training, 14,892 validation and
14,898 test reports; the ACN overlap between splits is zero.

The CSVs are not versioned here because of their size. Place them in `data/`
before running any script, or copy them from another location:

```bash
python scripts/01_prepare.py --copy /path/to/cleaned_data
```

## Model checkpoints

The stacking experiment reuses already trained Transformer classifiers. It
expects the following directories under `models/`, each containing a
HuggingFace checkpoint (`config.json`, weights, tokenizer files):

```
models/classifier          models/exp_a           models/exp_b_bert
models/exp_c_w2            models/exp_d_triw      models/exp_e_roberta
models/exp_f_albert        models/exp_g_focal
models/binary/bin_HumanFactors   models/binary/bin_Aircraft
models/binary/bin_Ambiguous      models/binary/bin_Procedure
models/binary/bin_CompanyPolicy  models/binary/bin_Weather
```

Checkpoints are not versioned here either. Any missing member can be produced
with `scripts/02_train_classifier.py` (single-label) and
`scripts/04_binary_per_class.py` (one-vs-rest), using the configuration in
`asrs_lib/config.py`. Scripts skip members whose directory is absent, so a
partial pool still runs and reports which members were used.

## Installation

```bash
python -m pip install -r requirements.txt
```

The pinned versions correspond to the environment in which the reported
experiments were executed (see `environment.md`). A GPU is recommended for
inference; a CPU-only run is possible but slow.

## Running

```bash
# 1. validate data and check split leakage
python scripts/01_prepare.py

# 2. train any missing member (example: the DistilBERT reference model)
python scripts/02_train_classifier.py --base-model distilbert-base-uncased \
       --output-dir models/classifier

# 3. six one-vs-rest binary classifiers (one positive-class weight per class)
python scripts/04_binary_per_class.py

# 4. extended ensembles and stacking (84 features, Logistic Regression)
python scripts/06_stack_ensemble.py

# 5. optional: bias tuning of the averaged pool and alternative metrics
python scripts/07_bias_tune_all.py
python scripts/08_metric_weights.py

# 6. optional: per-member metrics
python scripts/09_member_metrics.py
```

`run_all.ps1` executes the same sequence on Windows.

## Outputs

```
outputs/results/stack_ensemble_report.txt   ← per-class P/R/F1 and overall metrics
outputs/results/stack_probs/*.npy            ← cached val/test probabilities per member
outputs/results/data_report.txt              ← split sizes and leakage check
```

The probability cache makes the stacking, bias-tuning and metric-weighting
experiments reproducible without re-running inference: delete
`outputs/results/stack_probs/` to force recomputation.

## Stacking configuration

| Item | Value |
|---|---|
| Single-label members | 8 models, 6 probabilities each (48 features) |
| Binary members | 6 models, converted to 6 pseudo-probabilities each (36 features) |
| Total feature dimension | 84 |
| Meta-learner | Logistic Regression, L2, `C = 1.0`, `max_iter = 3000`, lbfgs |
| Fitting data | validation split only (14,892 reports) |
| Evaluation | test split (14,898 reports), never used for fitting or selection |

The binary pseudo-probability is constructed as `p` for the positive class and
`(1 - p) / 5` for each remaining class, so every binary member contributes a
six-dimensional vector consistent with the class ordering in
`asrs_lib/config.py`.

## Reproducibility notes

- A single global seed (`SEED = 42`) is used for training and for the
  meta-learner.
- The train/validation/test split is fixed and never regenerated.
- The meta-learner is fitted on validation probabilities only; the test split
  is used exclusively for the final evaluation.
- cuDNN deterministic flags were not enabled in the original runs, so bitwise
  reproducibility across different hardware is not guaranteed.
- Reported numbers were produced with maximum sequence length 256 for both
  training and inference; keep `MAX_LEN = 256` in `asrs_lib/config.py` when
  reproducing them.
- Some scripts keep historical step labels (for example "Step 18") in their
  docstrings; these refer to the original experiment log and do not affect
  execution order.
