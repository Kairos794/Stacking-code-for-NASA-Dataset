#!/usr/bin/env python3
"""
Step 2 - Fine-tune the sequence classifier (primary-problem 6-class task).

Adapted from the `ASRS_Classifier` GitHub repo for the 6 cleaned labels.
Train on GPU for best results; a fast CPU dry-run is available with ``--subset``.

Usage (GPU, full data):
    python 02_train_classifier.py

Fast CPU dry-run (few thousand samples, few epochs):
    python 02_train_classifier.py --subset 4000 --epochs 2

Notes:
  * Class-weighted loss is enabled by default to counter class imbalance.
  * The best model is selected on macro-F1 over the validation split.
  * Model + tokenizer are saved to models/classifier/.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asrs_lib import config, classifier, data_utils  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Train ASRS 6-class classifier")
    ap.add_argument("--subset", type=int, default=0,
                    help="stratified down-sample of the train set (0 = full)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--base-model", type=str, default=None)
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--no-class-weights", action="store_true",
                    help="disable inverse-frequency loss weighting")
    ap.add_argument("--weight-power", type=float, default=1.0,
                    help="sharpen class weights: (1/freq)^power (default 1.0)")
    ap.add_argument("--custom-weights", type=float, nargs=6, default=None,
                    help="6 explicit class weights in LABELS order "
                         "(overrides frequency-based weighting)")
    ap.add_argument("--loss", type=str, default="ce",
                    choices=["ce", "focal"],
                    help="loss function: ce (class-weighted cross entropy) "
                         "or focal (focal loss, GermEval approach)")
    ap.add_argument("--focal-gamma", type=float, default=1.0,
                    help="focal loss gamma (GermEval: 0.5-1.5 works best)")
    args = ap.parse_args()

    print("=" * 65)
    print("Training ASRS primary-problem classifier (6 classes)")
    print("=" * 65)
    splits = data_utils.load_all_splits()

    weights = None
    if config.CLASS_WEIGHTING and not args.no_class_weights:
        if args.custom_weights is not None:
            weights = list(args.custom_weights)
        else:
            weights = data_utils.class_weights_from(splits["train"],
                                                    power=args.weight_power)
        print("class weights:", {config.ID_TO_LABEL[i]: round(w, 3)
                                 for i, w in enumerate(weights)})

    t0 = time.time()
    model_dir, acc, f1 = classifier.train_classifier(
        train_texts=splits["train"]["text"].tolist(),
        train_labels=splits["train"]["label_id"].tolist(),
        val_texts=splits["val"]["text"].tolist(),
        val_labels=splits["val"]["label_id"].tolist(),
        class_weights=weights,
        base_model=args.base_model,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        subset=args.subset,
        max_len=args.max_len,
        loss_type=args.loss,
        focal_gamma=args.focal_gamma,
    )
    print(f"\n[done] model={model_dir} val_accuracy={acc:.4f} "
          f"val_macro_f1={f1:.4f} time={time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
