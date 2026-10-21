#!/usr/bin/env python3
"""
Step 18 - Extended ensemble + stacking on top of ALL existing models.

Two previously-untried ideas, zero training cost:
  1. PROB_AVG with an extended pool: all 8 single-label models PLUS the 6
     per-class binary classifiers (v1.10) as additional voters.
  2. Stacking: concat every model's val probabilities (8*6 + 6 = 54 features)
     and train a multinomial LogisticRegression (+ HistGBDT) on VAL to make the
     final decision, then evaluate on TEST.

Inference results are cached as .npy under outputs/results/stack_probs/ so
re-runs are instant.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asrs_lib import config, classifier, data_utils  # noqa: E402

SINGLE_MODELS = [
    "classifier", "exp_a", "exp_b_bert", "exp_c_w2",
    "exp_d_triw", "exp_e_roberta", "exp_f_albert", "exp_g_focal",
]
BINARY_MODELS = [f"bin_{n}" for n in
                 ("HumanFactors", "Aircraft", "Ambiguous", "Procedure",
                  "CompanyPolicy", "Weather")]
CACHE = config.RESULT_DIR / "stack_probs"


def _probs(model, tok, texts, device):
    return classifier.softmax_logits(model, tok, texts, device=device,
                                     batch_size=64)


def _binary_pseudo(probs, class_idx):
    """Turn a (n,2) binary softmax into a (n,6) pseudo probability vector."""
    p = probs[:, 1]
    out = np.empty((len(p), len(config.LABELS)))
    out[:] = (1 - p[:, None]) / (len(config.LABELS) - 1)
    out[:, class_idx] = p
    return out


def _load_cached_or_compute(model_dir, split_name, texts, device, class_idx=None):
    fname = CACHE / f"{Path(model_dir).name}__{split_name}.npy"
    if fname.exists():
        return np.load(fname)
    model, tok = classifier.load_classifier(model_dir)
    if Path(model_dir).name.startswith("bin_"):
        probs = _binary_pseudo(_probs(model, tok, texts, device), class_idx)
    else:
        probs = _probs(model, tok, texts, device)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(fname, probs)
    return probs


def main() -> int:
    ap = argparse.ArgumentParser(description="Extended ensemble + stacking")
    ap.add_argument("--no-single", action="store_true")
    ap.add_argument("--no-binary", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    print("=" * 65)
    print("Extended ensemble + stacking (all models)")
    print("=" * 65)
    splits = data_utils.load_all_splits()
    val_df, test_df = splits["val"], splits["test"]
    y_val, y_test = val_df["label_id"].to_numpy(), test_df["label_id"].to_numpy()
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if args.no_cache:
        CACHE.mkdir(parents=True, exist_ok=True)

    single_dirs = [config.MODEL_DIR / m for m in SINGLE_MODELS
                   if (config.MODEL_DIR / m / "config.json").exists()]
    bin_dirs = [config.MODEL_DIR / "binary" / m for m in BINARY_MODELS
                if (config.MODEL_DIR / "binary" / m / "config.json").exists()]
    if args.no_single:
        single_dirs = []
    if args.no_binary:
        bin_dirs = []
    print(f"single={[d.name for d in single_dirs]} "
          f"binary={[d.name for d in bin_dirs]}")

    t0 = time.time()
    s_val = [(_load_cached_or_compute(d, "val", val_df["text"].tolist(), dev),
              d.name) for d in single_dirs]
    s_test = [(_load_cached_or_compute(d, "test", test_df["text"].tolist(), dev),
               d.name) for d in single_dirs]
    b_val = [(_load_cached_or_compute(d, "val", val_df["text"].tolist(), dev,
                                      class_idx=BINARY_MODELS.index(d.name)),
              d.name) for d in bin_dirs]
    b_test = [(_load_cached_or_compute(d, "test", test_df["text"].tolist(), dev,
                                       class_idx=BINARY_MODELS.index(d.name)),
               d.name) for d in bin_dirs]
    print(f"[infer] done in {time.time()-t0:.0f}s")

    lines = []
    lines.append(f"single={[d.name for d in single_dirs]} "
                 f"binary={[d.name for d in bin_dirs]}")

    def report(tag, prob_test, y=y_test):
        preds = np.argmax(prob_test, axis=1)
        acc = accuracy_score(y, preds)
        f1 = f1_score(y, preds, average="macro", zero_division=0)
        per = precision_recall_fscore_support(
            y, preds, labels=list(range(len(config.LABELS))), zero_division=0)
        lines.append(f"[{tag}] acc={acc:.4f} macro_F1={f1:.4f}")
        return per, preds

    # ---- PROB_AVG extended -------------------------------------------------
    if s_test and b_test:
        avg_all = np.mean([p for p, _ in s_test + b_test], axis=0)
        _, preds_all = report("PROB_AVG all", avg_all)
    if s_test:
        avg_s = np.mean([p for p, _ in s_test], axis=0)
        _, preds_s = report("PROB_AVG single", avg_s)
    if b_test:
        avg_b = np.mean([p for p, _ in b_test], axis=0)
        _, preds_b = report("PROB_AVG binary", avg_b)

    # ---- Stacking: LR on val, evaluated on test ----------------------------
    if s_val and b_val:
        X_val = np.hstack([p for p, _ in s_val + b_val])
        X_test = np.hstack([p for p, _ in s_test + b_test])
        lr = LogisticRegression(C=1.0, max_iter=3000)
        lr.fit(X_val, y_val)
        lr_per, _ = report("STACK LR", lr.predict_proba(X_test))
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier
            hgb = HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.08, max_depth=4,
                early_stopping=True, validation_fraction=0.2,
                random_state=42)
            hgb.fit(X_val, y_val)
            report("STACK HistGBDT", hgb.predict_proba(X_test))
        except Exception as e:  # noqa: BLE001
            lines.append(f"[STACK HistGBDT] skipped: {e}")

    # ---- per-class F1 for the stacking LR --------------------------------
    if s_val and b_val:
        per = lr_per
        lines.append(f"{'class':<16} {'P':>6} {'R':>6} {'F1':>6}")
        for i, label in enumerate(config.LABELS):
            lines.append(f"{label:<16} {per[0][i]:>6.3f} {per[1][i]:>6.3f} "
                         f"{per[2][i]:>6.3f}")

    report_txt = "\n".join(lines)
    print("\n" + report_txt)
    out_path = config.RESULT_DIR / "stack_ensemble_report.txt"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(report_txt + "\n")
    print(f"[report] -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
