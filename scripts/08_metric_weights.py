#!/usr/bin/env python3
"""
Step 20 - Metric weighting (option 1): recompute evaluation under several
class-weighting schemes to counter the HF-absorber / tiny-class drag.

No retraining. Uses cached per-model probs (Step 18) for the 14-voter
PROB_AVG and re-fits the stacking LR, then reports:
  - macro-F1            (per-class unweighted average)
  - weighted-F1         (weighted by true sample counts)
  - custom-weighted F1  (user-style weights, e.g. down-weight the ambiguous
                         catch-all class, up-weight tiny classes)
  - per-class F1 detail for transparency
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asrs_lib import config, data_utils  # noqa: E402

CACHE = config.RESULT_DIR / "stack_probs"
SINGLE = ["classifier", "exp_a", "exp_b_bert", "exp_c_w2", "exp_d_triw",
          "exp_e_roberta", "exp_f_albert", "exp_g_focal"]
BINARY = [f"bin_{n}" for n in
          ("HumanFactors", "Aircraft", "Ambiguous", "Procedure",
           "CompanyPolicy", "Weather")]


def avg(name_list, split):
    mats = [np.load(CACHE / f"{n}__{split}.npy") for n in name_list]
    return np.mean(mats, axis=0)


def weighted_f1(y, preds, weights):
    """Weighted macro-F1 with arbitrary per-class weights (normalised)."""
    f1s = f1_score(y, preds, average=None, zero_division=0)
    w = np.asarray(weights, float)
    w = w / w.sum()
    return float(np.sum(w * f1s))


def main() -> int:
    splits = data_utils.load_all_splits()
    val_df, test_df = splits["val"], splits["test"]
    y_val, y_test = val_df["label_id"].to_numpy(), test_df["label_id"].to_numpy()

    av_test = avg(SINGLE + BINARY, "test")
    preds_avg = np.argmax(av_test, axis=1)

    # stacking LR predictions (same as Step 18)
    X_val = np.hstack([np.load(CACHE / f"{n}__val.npy") for n in SINGLE + BINARY])
    X_test = np.hstack([np.load(CACHE / f"{n}__test.npy") for n in SINGLE + BINARY])
    lr = LogisticRegression(C=1.0, max_iter=3000).fit(X_val, y_val)
    preds_lr = lr.predict(X_test)

    dist = np.bincount(y_test, minlength=len(config.LABELS))
    dist_norm = dist / dist.sum()

    # custom weights: de-emphasise HF (absorber) and Ambiguous (catch-all),
    # slightly boost the remaining tiny classes -> "problem-relevant macro".
    custom = np.array([0.6, 1.0, 0.7, 1.2, 1.2, 1.2])

    lines = [f"{'scheme':<30} {'PROB_AVG all':>14} {'STACK LR':>14}",
             f"{'macro-F1 (per-class avg)':<30} "
             f"{f1_score(y_test, preds_avg, average='macro', zero_division=0):>14.4f} "
             f"{f1_score(y_test, preds_lr, average='macro', zero_division=0):>14.4f}",
             f"{'weighted-F1 (by counts)':<30} "
             f"{f1_score(y_test, preds_avg, average='weighted', zero_division=0):>14.4f} "
             f"{f1_score(y_test, preds_lr, average='weighted', zero_division=0):>14.4f}",
             f"{'custom-weighted F1':<30} "
             f"{weighted_f1(y_test, preds_avg, custom):>14.4f} "
             f"{weighted_f1(y_test, preds_lr, custom):>14.4f}",
             ""]
    lines.append(f"class dist (test): "
                 + ", ".join(f"{config.LABELS[i]}={dist_norm[i]:.2%}" for i in range(6)))
    lines.append(f"custom weights   : "
                 + ", ".join(f"{config.LABELS[i]}={custom[i]:.1f}" for i in range(6)))
    lines.append("")
    lines.append(f"{'class':<16} {'support':>8} {'PROB_AVG F1':>11} {'STACK LR F1':>11}")
    f1a = f1_score(y_test, preds_avg, average=None, zero_division=0)
    f1l = f1_score(y_test, preds_lr, average=None, zero_division=0)
    for i in range(6):
        lines.append(f"{config.LABELS[i]:<16} {dist[i]:>8d} {f1a[i]:>11.3f} "
                     f"{f1l[i]:>11.3f}")

    report = "\n".join(lines)
    print(report)
    with open(config.RESULT_DIR / "metric_weights_report.txt", "w",
              encoding="utf-8") as fh:
        fh.write(report + "\n")
    print(f"[report] -> {config.RESULT_DIR / 'metric_weights_report.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
