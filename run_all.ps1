# Run the stacking pipeline from the repository root.
#
# Usage:
#   .\run_all.ps1              # assume checkpoints exist under models/
#   .\run_all.ps1 -TrainMissing # first train any missing members
#
param(
    [switch]$TrainMissing
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

if ($TrainMissing) {
    Write-Host "[run_all] training missing single-label members"
    python scripts/02_train_classifier.py --base-model distilbert-base-uncased --output-dir models/classifier
    python scripts/02_train_classifier.py --base-model bert-base-uncased       --output-dir models/exp_b_bert
    python scripts/02_train_classifier.py --base-model roberta-base            --output-dir models/exp_e_roberta
    python scripts/02_train_classifier.py --base-model albert-base-v2          --output-dir models/exp_f_albert
    python scripts/02_train_classifier.py --base-model distilbert-base-uncased --custom-weights 1 1 4 4 5 5 --output-dir models/exp_d_triw

    Write-Host "[run_all] training one-vs-rest binary classifiers"
    python scripts/04_binary_per_class.py
}

Write-Host "[run_all] validating data"
python scripts/01_prepare.py

Write-Host "[run_all] stacking (84 features, Logistic Regression)"
python scripts/06_stack_ensemble.py

Write-Host "[run_all] optional analyses"
python scripts/07_bias_tune_all.py
python scripts/08_metric_weights.py
python scripts/09_member_metrics.py

Write-Host "[run_all] finished; see outputs/results/"
