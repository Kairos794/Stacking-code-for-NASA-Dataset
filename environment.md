# Environment record

All values below were read from the machine on which the reported experiments
were executed. No entry is estimated.

## Software environment

| Component | Version |
|---|---|
| Operating system | Microsoft Windows 10 Pro, build 10.0.19045 (64-bit) |
| Python | 3.14.0 |
| PyTorch | 2.11.0+cu128 |
| CUDA runtime (torch) | 12.8 |
| cuDNN | 9.19.0 |
| HuggingFace Transformers | 5.16.1 |
| tokenizers | 0.23.1 |
| datasets | 5.0.1 |
| accelerate | 1.15.0 |
| safetensors | 0.8.0 |
| NumPy | 2.5.2 |
| Pandas | 3.0.5 |
| SciPy | 1.18.0 |
| Scikit-learn | 1.9.0 |
| Matplotlib | 3.11.1 |
| psutil | 7.2.2 |

`sentencepiece` and `seqeval` were not installed: ALBERT uses the official
`tokenizer.json`, and all metrics are computed with `sklearn.metrics`.

## Hardware environment

| Item | Value |
|---|---|
| Platform | Windows 10 Pro workstation (local machine, not a Linux server) |
| CPU | 2 × AMD EPYC 7763, 128 physical cores / 256 threads, 2.45 GHz |
| System RAM | 384 GB |
| GPU | 1 × NVIDIA GeForce RTX 3090 |
| GPU memory | 24 GB |
| GPU driver | 591.44 |
| Compute capability | sm_86 |
| GPU count | single GPU; all experiments ran serially on it |

No DeepSpeed, FSDP or multi-GPU parallelisation was used. Training used
single-GPU FP16 automatic mixed precision.

## Training configuration (per member)

| Setting | Value |
|---|---|
| Maximum sequence length | 256 (training and inference) |
| Per-device batch size | 16 (training and evaluation) |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Epochs | 4 |
| Warm-up | ratio 0.06 (1,042 steps) |
| Optimiser | AdamW (`adamw_torch`), no gradient accumulation |
| Precision | FP16 |
| Label smoothing / early stopping | not used |
| Checkpoint selection | best validation Macro-F1, epoch-level checkpoints |
| Random seed | 42 |
| Loss | class-weighted cross-entropy |

Inverse-frequency class weights (power 1) used by the standard members:
`[0.8214, 0.8820, 1.0436, 1.0451, 1.0955, 1.1123]`. The tri-weight member uses
the fixed vector `[1, 1, 4, 4, 5, 5]`.

## Ensemble and stacking configuration

| Item | Value |
|---|---|
| Five-model ensemble | probability averaging, majority vote, class-specialised weighting |
| Inference batch size | 64 |
| Stacking features | 84 = 8 single-label models × 6 classes + 6 binary models × 6 pseudo-probabilities |
| Meta-learner | Logistic Regression, lbfgs, L2, `C = 1.0`, `max_iter = 3000`, `tol = 1e-4` |
| Fitting data | validation split only (14,892 reports) |
| Evaluation split | fixed test split (14,898 reports) |

## Reproducibility controls

| Item | Value |
|---|---|
| Data split | fixed stratified 70/15/15 CSVs, never regenerated |
| Seed | single global seed 42 |
| Determinism | cuDNN deterministic flags not set (noted as a limitation) |
| Probability cache | `outputs/results/stack_probs/*.npy`, enables exact recomputation |
| Logs and results | `outputs/logs/*.log`, `outputs/results/*.txt`, `models/*/checkpoints/*/trainer_state.json` |
