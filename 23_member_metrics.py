#!/usr/bin/env python3
"""Step 23 - Compile per-member test metrics from cached probs (backfill log)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asrs_lib import config, data_utils  # noqa: E402

CACHE = config.RESULT_DIR / "stack_probs"
SINGLE = ["classifier", "exp_a", "exp_b_bert", "exp_c_w2",
          "exp_d_triw", "exp_e_roberta", "exp_f_albert", "exp_g_focal"]
BINARY = [f"bin_{n}" for n in
          ("HumanFactors", "Aircraft", "Ambiguous", "Procedure",
           "CompanyPolicy", "Weather")]


def main() -> int:
    splits = data_utils.load_all_splits()
    y_test = splits["test"]["label_id"].to_numpy()
    lines = []
    for name in SINGLE + BINARY:
        p = np.load(CACHE / f"{name}__test.npy")
        preds = np.argmax(p, axis=1)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro", zero_division=0)
        per = precision_recall_fscore_support(
            y_test, preds, labels=list(range(len(config.LABELS))),
            zero_division=0)
        lines.append(f"{name} acc={acc:.4f} macroF1={f1:.4f}")
        f1s = " ".join(f"{v:.3f}" for v in per[2])
        lines.append(f"  F1[{','.join(config.LABELS)}] {f1s}")
    report = "\n".join(lines)
    print(report)
    with open(config.RESULT_DIR / "member_metrics.txt", "w",
              encoding="utf-8") as fh:
        fh.write(report + "\n")
    print(f"[report] -> {config.RESULT_DIR / 'member_metrics.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
