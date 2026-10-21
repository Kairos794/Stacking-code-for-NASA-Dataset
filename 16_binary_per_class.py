#!/usr/bin/env python3
"""
Step 16 - Per-class independent binary classifiers (SafeAeroBERT replication).

Replicates the task design used by NASA's SafeAeroBERT / DeBERTa-AT: instead of
one 6-way softmax, train one binary classifier per class ("class present vs
not"), then combine their positive scores into a decision.

Two evaluations are reported on the SAME fixed split as the 6-class baseline:
  1. Per-class binary metrics (F1/P/R of each binary model) - matches how the
     NASA papers report per-class performance.
  2. Combined single-label metrics (argmax over the 6 positive scores) with an
     optional per-class bias sweep tuned on val and applied to test - matches
     our project rubric (accuracy + macro-F1).

Usage (GPU, full data):
    python 16_binary_per_class.py

Fast dry-run:
    python 16_binary_per_class.py --classes 3,5 --epochs 2 --subset 6000

Reuse already trained per-class models (skip training):
    python 16_binary_per_class.py --eval-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Force local-only loading of the (already cached) distilbert checkpoint.
# Without this, newer transformers/huggingface_hub contact the Hub even for
# cached models and abort training on a network timeout (seen as
# httpx.ConnectTimeout in build_tokenizer).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asrs_lib import config, classifier, data_utils  # noqa: E402

# Sanitised per-class model directory names (config.LABELS order).
MODEL_NAME = {
    "Human Factors": "bin_HumanFactors",
    "Aircraft": "bin_Aircraft",
    "Ambiguous": "bin_Ambiguous",
    "Procedure": "bin_Procedure",
    "Company Policy": "bin_CompanyPolicy",
    "Weather": "bin_Weather",
}


def _bin_dir(class_idx: int) -> Path:
    return config.MODEL_DIR / "binary" / MODEL_NAME[config.ID_TO_LABEL[class_idx]]


def _positive_prob(model, tokenizer, texts, device):
    """Return per-sample probability of binary label 1 (the class present)."""
    probs = classifier.softmax_logits(model, tokenizer, texts, device=device)
    return probs[:, 1]


def _binary_class_weights(train_df, class_idx: int, power: float):
    """Loss weights [neg, pos] for one class vs rest.

    pos weight = (neg/pos)^power so rare classes get a stronger positive
    gradient (power=0.5 = square-root inverse frequency, a middle ground).
    """
    pos = int((train_df["label_id"] == class_idx).sum())
    neg = len(train_df) - pos
    if power <= 0 or pos == 0:
        return None
    return [1.0, float((neg / pos) ** power)]


def _combined_macro_f1(p_matrix, y_true):
    """argmax over per-class positive scores -> single-label metrics."""
    preds = np.argmax(p_matrix, axis=1)
    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, average="macro", zero_division=0)
    per_class = precision_recall_fscore_support(
        y_true, preds, labels=list(range(len(config.LABELS))),
        zero_division=0,
    )[:3]
    return acc, f1, per_class


def _tune_biases(p_val, y_val, p_test, y_test):
    """Coordinate-ascent per-class log-bias tuned on val, applied to test.

    score_c = log(p_c) + bias_c ; prediction = argmax_c score_c. Tuning these
    per-class biases is the "per-class decision threshold" of the binary
    approach, optimised for macro-F1 on validation only.
    """
    n_classes = p_val.shape[1]
    p_clip = np.clip(p_val, 1e-6, 1 - 1e-6)
    t_clip = np.clip(p_test, 1e-6, 1 - 1e-6)
    biases = np.zeros(n_classes)
    best_f1 = f1_score(y_val, np.argmax(np.log(p_clip), axis=1),
                       average="macro", zero_division=0)
    grid = np.arange(-2.0, 2.01, 0.1)
    for _ in range(3):  # a few full passes
        for c in range(n_classes):
            cand = []
            for b in grid:
                trial = biases.copy()
                trial[c] = b
                preds = np.argmax(np.log(p_clip) + trial[None, :], axis=1)
                cand.append((b, f1_score(y_val, preds, average="macro",
                                         zero_division=0)))
            best_b, best_b_f1 = max(cand, key=lambda x: x[1])
            if best_b_f1 > best_f1 + 1e-6:
                biases[c] = best_b
                best_f1 = best_b_f1
    preds = np.argmax(np.log(t_clip) + biases[None, :], axis=1)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro", zero_division=0)
    per_class = precision_recall_fscore_support(
        y_test, preds, labels=list(range(len(config.LABELS))),
        zero_division=0,
    )[:3]
    return biases, acc, f1, per_class


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-class binary classifiers")
    ap.add_argument("--classes", type=str, default="0,1,2,3,4,5",
                    help="comma list of class ids to train (default all)")
    ap.add_argument("--base-model", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--weight-power", type=float, default=0.5,
                    help="positive-class weight exponent (0 disables weighting)")
    ap.add_argument("--eval-only", action="store_true",
                    help="skip training, evaluate existing binary models")
    ap.add_argument("--no-tune", action="store_true",
                    help="skip the per-class bias sweep (argmax only)")
    args = ap.parse_args()

    class_ids = [int(c) for c in args.classes.split(",") if c.strip()]
    for c in class_ids:
        if c < 0 or c >= len(config.LABELS):
            raise SystemExit(f"bad class id {c}; LABELS={config.LABELS}")

    print("=" * 65)
    print("Per-class binary classifiers (SafeAeroBERT replication)")
    print("=" * 65)
    splits = data_utils.load_all_splits()
    train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]
    y_val = val_df["label_id"].to_numpy()
    y_test = test_df["label_id"].to_numpy()
    import torch
    use_cuda = torch.cuda.is_available()

    # ---- 1. Train one binary classifier per class -------------------------
    if not args.eval_only:
        for c in class_ids:
            label = config.ID_TO_LABEL[c]
            mdir = _bin_dir(c)
            if (mdir / "config.json").exists():
                print(f"[skip] {label}: model already exists at {mdir}")
                continue
            print(f"\n--- binary model for class [{c}] {label} ---")
            y_bin_tr = (train_df["label_id"] == c).astype(int).tolist()
            y_bin_va = (val_df["label_id"] == c).astype(int).tolist()
            weights = _binary_class_weights(train_df, c, args.weight_power)
            if weights is not None:
                print(f"    binary weights [neg, pos]={weights}")
            t0 = time.time()
            out_dir, acc, f1 = classifier.train_classifier(
                train_texts=train_df["text"].tolist(),
                train_labels=y_bin_tr,
                val_texts=val_df["text"].tolist(),
                val_labels=y_bin_va,
                class_weights=weights,
                base_model=args.base_model,
                output_dir=_bin_dir(c),
                epochs=args.epochs,
                batch_size=args.batch_size,
                subset=args.subset,
                max_len=args.max_len,
                num_labels=2,
                id2label={0: "Other", 1: label},
                label2id={"Other": 0, label: 1},
            )
            print(f"[done] {label}: {out_dir} val_acc={acc:.4f} "
                  f"val_macro_f1={f1:.4f} time={time.time()-t0:.1f}s")

    # ---- 2. Evaluate on val + test ----------------------------------------
    P_val = np.zeros((len(val_df), len(config.LABELS)))
    P_test = np.zeros((len(test_df), len(config.LABELS)))
    binary_metrics = {}
    trained_ids = []
    for c in range(len(config.LABELS)):
        label = config.ID_TO_LABEL[c]
        mdir = _bin_dir(c)
        if not (mdir / "config.json").exists():
            print(f"[eval] skip class {label}: model dir {mdir} missing")
            continue
        trained_ids.append(c)
        model, tok = classifier.load_classifier(mdir)
        P_val[:, c] = _positive_prob(model, tok, val_df["text"].tolist(),
                                     device="cuda" if use_cuda else "cpu")
        P_test[:, c] = _positive_prob(model, tok, test_df["text"].tolist(),
                                      device="cuda" if use_cuda else "cpu")
        bpred = P_test[:, c] >= 0.5
        p, r, f, _ = precision_recall_fscore_support(
            (test_df["label_id"] == c).astype(int).to_numpy(), bpred,
            average="binary", zero_division=0)
        binary_metrics[label] = (p, r, f)
        print(f"[eval] binary {label}: P={p:.4f} R={r:.4f} F1={f:.4f}")

    lines = []
    lines.append("Per-class binary classifiers (SafeAeroBERT replication)")
    lines.append(f"classes trained: {[config.ID_TO_LABEL[c] for c in trained_ids]}")
    lines.append("")
    lines.append("== (1) Per-class binary metrics on test (NASA-paper style) ==")
    lines.append(f"{'class':<16} {'P':>6} {'R':>6} {'F1':>6}")
    for label, (p, r, f) in binary_metrics.items():
        lines.append(f"{label:<16} {p:>6.3f} {r:>6.3f} {f:>6.3f}")

    complete = len(trained_ids) == len(config.LABELS)
    if complete and P_val.any() and P_test.any():
        lines.append("")
        lines.append("== (2) Combined single-label decision on test ==")
        acc, f1, per_class = _combined_macro_f1(P_test, y_test)
        lines.append(f"[argmax ] accuracy={acc:.4f} macro_F1={f1:.4f}")
        if not args.no_tune and complete:
            biases, tacc, tf1, tper = _tune_biases(
                P_val, y_val, P_test, y_test)
            lines.append(f"[tuned  ] accuracy={tacc:.4f} macro_F1={tf1:.4f}")
            lines.append(f"biases={np.round(biases, 2).tolist()}")
            lines.append(f"{'class':<16} {'P':>6} {'R':>6} {'F1':>6}")
            for i, label in enumerate(config.LABELS):
                lines.append(f"{label:<16} {tper[0][i]:>6.3f} "
                             f"{tper[1][i]:>6.3f} {tper[2][i]:>6.3f}")

    report = "\n".join(lines)
    print("\n" + report)
    out_path = config.RESULT_DIR / "binary_per_class_report.txt"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    print(f"[report] -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
