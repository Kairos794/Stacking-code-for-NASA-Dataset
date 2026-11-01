#!/usr/bin/env python3
"""Step 19 - Bias-tune the 14-voter PROB_AVG (v1.12 pool) on val, eval on test.

Reads the cached per-model probabilities from Step 18, averages all voters,
then runs coordinate-ascent per-class log-bias tuning on VAL (macro-F1
objective) and applies the frozen biases to TEST. Zero-cost re-run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asrs_lib import config, data_utils  # noqa: E402

CACHE = config.RESULT_DIR / "stack_probs"
SINGLE = ["classifier", "exp_a", "exp_b_bert", "exp_c_w2", "exp_d_triw",
          "exp_e_roberta", "exp_f_albert", "exp_g_focal"]
BINARY = [f"bin_{n}" for n in
          ("HumanFactors", "Aircraft", "Ambiguous", "Procedure",
           "CompanyPolicy", "Weather")]
EPS = 1e-9


def _avg(name_list, split):
    mats = [np.load(CACHE / f"{n}__{split}.npy") for n in name_list]
    return np.mean(mats, axis=0)


def _tune(p_val, y_val, p_test, y_test):
    p_val = np.clip(p_val, EPS, 1.0)
    p_test = np.clip(p_test, EPS, 1.0)
    lv, lt = np.log(p_val), np.log(p_test)
    n = p_val.shape[1]
    bias = np.zeros(n)
    best_f1 = f1_score(y_val, np.argmax(lv, axis=1), average="macro",
                       zero_division=0)
    grid = np.arange(-1.5, 1.51, 0.1)
    for _ in range(3):
        for c in range(n):
            best_b, best_b_f1 = None, best_f1
            for b in grid:
                tb = bias.copy()
                tb[c] = b
                f = f1_score(y_val, np.argmax(lv + tb[None, :], axis=1),
                             average="macro", zero_division=0)
                if f > best_b_f1 + 1e-6:
                    best_b, best_b_f1 = b, f
            if best_b is not None:
                bias[c] = best_b
                best_f1 = best_b_f1
    preds = np.argmax(lt + bias[None, :], axis=1)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro", zero_division=0)
    per = precision_recall_fscore_support(
        y_test, preds, labels=list(range(len(config.LABELS))), zero_division=0)
    return bias, best_f1, acc, f1, per


def main() -> int:
    splits = data_utils.load_all_splits()
    val_df, test_df = splits["val"], splits["test"]
    y_val, y_test = val_df["label_id"].to_numpy(), test_df["label_id"].to_numpy()

    av_val = _avg(SINGLE + BINARY, "val")
    av_test = _avg(SINGLE + BINARY, "test")

    # baseline before tuning
    p0 = np.argmax(av_test, axis=1)
    print(f"[before] acc={accuracy_score(y_test, p0):.4f} "
          f"macro_F1={f1_score(y_test, p0, average='macro', zero_division=0):.4f}")

    bias, val_f1, acc, f1, per = _tune(av_val, y_val, av_test, y_test)
    lines = [f"14-voter PROB_AVG + val bias tuning",
             f"[val] macro_F1={val_f1:.4f}",
             f"[test] accuracy={acc:.4f} macro_F1={f1:.4f}",
             f"biases={np.round(bias, 2).tolist()}",
             f"{'class':<16} {'P':>6} {'R':>6} {'F1':>6}"]
    for i, label in enumerate(config.LABELS):
        lines.append(f"{label:<16} {per[0][i]:>6.3f} {per[1][i]:>6.3f} "
                     f"{per[2][i]:>6.3f}")
    report = "\n".join(lines)
    print("\n" + report)
    with open(config.RESULT_DIR / "bias_tune_all_report.txt", "w",
              encoding="utf-8") as fh:
        fh.write(report + "\n")
    print(f"[report] -> {config.RESULT_DIR / 'bias_tune_all_report.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
