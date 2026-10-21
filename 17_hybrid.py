#!/usr/bin/env python3
"""
Step 17 - Per-class source hybrid (6-class softmax + per-class binary).

Idea from user: use the per-class binary classifiers for some classes and the
original 6-class softmax for the others, then combine into one single-label
decision. This script builds, per class, a score vector whose c-th dimension
comes from ONE of two sources:
    single: log-softmax of the 6-class model (default exp_f_albert)
    bin   : log positive-prob of the per-class binary classifier (v1.10)

Then it optimises per-class (source, bias) on VAL by coordinate ascent to
maximise val macro-F1, freezes it, and evaluates on TEST. It also reports the
fixed assignment the user proposed (bin for Human Factors + Aircraft, single
for the rest) for direct comparison.

Zero training cost - reuses existing checkpoints.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from asrs_lib import config, classifier, data_utils  # noqa: E402

BIN_DIR = config.MODEL_DIR / "binary"
BIN_NAME = {
    0: "bin_HumanFactors", 1: "bin_Aircraft", 2: "bin_Ambiguous",
    3: "bin_Procedure", 4: "bin_CompanyPolicy", 5: "bin_Weather",
}
EPS = 1e-9


def _log_probs_single(model, tok, texts, device):
    p = classifier.softmax_logits(model, tok, texts, device=device, batch_size=64)
    return np.log(np.clip(p, EPS, 1.0))


def _log_probs_binary(model, tok, texts, device):
    p = classifier.softmax_logits(model, tok, texts, device=device, batch_size=64)
    return np.log(np.clip(p[:, 1], EPS, 1.0))


def _metrics(scores, y_true):
    preds = np.argmax(scores, axis=1)
    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, average="macro", zero_division=0)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, preds, labels=list(range(len(config.LABELS))), zero_division=0)
    return acc, f1, (p, r, f)


def _coordinate_ascent(logS, logB, y_val):
    """Optimise per-class (source, bias) to maximise val macro-F1."""
    n = logS.shape[1]
    source = np.zeros(n, dtype=int)      # 0=single, 1=binary
    bias = np.zeros(n)
    grid = np.arange(-1.5, 1.51, 0.1)
    best_f1 = f1_score(y_val, np.argmax(logS, axis=1), average="macro",
                       zero_division=0)
    for _ in range(3):
        for c in range(n):
            best_cfg, best_c_f1 = None, best_f1
            for s in (0, 1):
                base = logS if s == 0 else logB
                for b in grid:
                    trial_s, trial_b = source.copy(), bias.copy()
                    trial_s[c], trial_b[c] = s, b
                    scores = base + trial_b[None, :]
                    preds = np.argmax(scores, axis=1)
                    f = f1_score(y_val, preds, average="macro", zero_division=0)
                    if f > best_c_f1 + 1e-6:
                        best_cfg, best_c_f1 = (s, b), f
            if best_cfg is not None:
                source[c], bias[c] = best_cfg
                best_f1 = best_c_f1
    return source, bias, best_f1


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-class source hybrid")
    ap.add_argument("--single-model", type=str, default="exp_f_albert",
                    help="6-class model dir under models/")
    ap.add_argument("--fixed-bin", type=str, default="0,1",
                    help="classes to force binary in the fixed user-assignment "
                         "(default 0,1 = HF,Aircraft)")
    ap.add_argument("--no-auto", action="store_true",
                    help="skip automatic val-optimised source selection")
    args = ap.parse_args()

    print("=" * 65)
    print("Per-class source hybrid (single softmax + per-class binary)")
    print("=" * 65)
    splits = data_utils.load_all_splits()
    val_df, test_df = splits["val"], splits["test"]
    y_val, y_test = val_df["label_id"].to_numpy(), test_df["label_id"].to_numpy()
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- load 6-class model ---------------------------------------------
    single_dir = config.MODEL_DIR / args.single_model
    if not (single_dir / "config.json").exists():
        raise SystemExit(f"single model not found: {single_dir}")
    model, tok = classifier.load_classifier(single_dir)
    print(f"[single] {single_dir} (val/test) inferring ...")
    S_val = _log_probs_single(model, tok, val_df["text"].tolist(), dev)
    S_test = _log_probs_single(model, tok, test_df["text"].tolist(), dev)

    # ---- load 6 binary models ------------------------------------------
    B_val = np.full((len(val_df), len(config.LABELS)), -20.0)
    B_test = np.full((len(test_df), len(config.LABELS)), -20.0)
    for c in range(len(config.LABELS)):
        mdir = BIN_DIR / BIN_NAME[c]
        if not (mdir / "config.json").exists():
            print(f"[bin] skip {BIN_NAME[c]} (missing)")
            continue
        bm, btok = classifier.load_classifier(mdir)
        B_val[:, c] = _log_probs_binary(bm, btok, val_df["text"].tolist(), dev)
        B_test[:, c] = _log_probs_binary(bm, btok, test_df["text"].tolist(), dev)
        print(f"[bin] {BIN_NAME[c]} loaded")

    lines = []
    lines.append(f"Per-class source hybrid | single={args.single_model}")
    lines.append("")

    # ---- baseline: pure single 6-class softmax --------------------------
    acc, f1, per = _metrics(S_test, y_test)
    lines.append(f"[pure single      ] acc={acc:.4f} macro_F1={f1:.4f}")
    lines.append(f"[pure single (val)] acc={accuracy_score(y_val, np.argmax(S_val, axis=1)):.4f} "
                 f"macro_F1={f1_score(y_val, np.argmax(S_val, axis=1), average='macro', zero_division=0):.4f}")

    # ---- pure binary argmax (v1.10 replication) -------------------------
    acc, f1, per = _metrics(B_test, y_test)
    lines.append(f"[pure binary      ] acc={acc:.4f} macro_F1={f1:.4f}")

    # ---- fixed user assignment: bin for {args.fixed-bin}, single elsewhere
    fixed_src = np.zeros(len(config.LABELS), dtype=int)
    for c in args.fixed_bin.split(","):
        c = int(c)
        if 0 <= c < len(config.LABELS):
            fixed_src[c] = 1
    # tune biases for this fixed assignment on val
    best_fixed_bias, best_fixed_f1 = np.zeros(len(config.LABELS)), -1
    grid = np.arange(-1.5, 1.51, 0.1)
    src_val = np.where(fixed_src[None, :] == 1, B_val, S_val)
    src_test = np.where(fixed_src[None, :] == 1, B_test, S_test)
    for _ in range(3):
        for c in range(len(config.LABELS)):
            best_b, best_b_f1 = None, best_fixed_f1
            for b in grid:
                tb = best_fixed_bias.copy()
                tb[c] = b
                preds = np.argmax(src_val + tb[None, :], axis=1)
                f = f1_score(y_val, preds, average="macro", zero_division=0)
                if f > best_b_f1 + 1e-6:
                    best_b, best_b_f1 = b, f
            if best_b is not None:
                best_fixed_bias[c] = best_b
                best_fixed_f1 = best_b_f1
    acc, f1, per = _metrics(src_test + best_fixed_bias[None, :], y_test)
    fixed_names = [config.ID_TO_LABEL[c] for c, s in enumerate(fixed_src) if s == 1]
    lines.append(f"[fixed user assign] bin={fixed_names} acc={acc:.4f} "
                 f"macro_F1={f1:.4f} (bias tuned on val)")

    # ---- automatic val-optimised (source, bias) -------------------------
    if not args.no_auto:
        src, bias, val_f1 = _coordinate_ascent(S_val, B_val, y_val)
        auto_val = np.where(src[None, :] == 1, B_val, S_val) + bias[None, :]
        auto_test = np.where(src[None, :] == 1, B_test, S_test) + bias[None, :]
        acc, f1, per = _metrics(auto_test, y_test)
        lines.append("")
        lines.append(f"[auto hybrid (val)] macro_F1={val_f1:.4f}")
        lines.append(f"[auto hybrid (test)] acc={acc:.4f} macro_F1={f1:.4f}")
        lines.append(f"chosen sources: "
                     + ", ".join(f"{config.ID_TO_LABEL[c]}={'bin' if s else 'single'}"
                                for c, s in enumerate(src))
                     + f" | biases={np.round(bias, 2).tolist()}")
        lines.append(f"{'class':<16} {'P':>6} {'R':>6} {'F1':>6}")
        for i, label in enumerate(config.LABELS):
            lines.append(f"{label:<16} {per[0][i]:>6.3f} {per[1][i]:>6.3f} {per[2][i]:>6.3f}")

    report = "\n".join(lines)
    print("\n" + report)
    out_path = config.RESULT_DIR / "hybrid_report.txt"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    print(f"[report] -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
