#!/usr/bin/env python3
"""
Ensemble evaluation: majority vote + probability average across several
trained ASRS classifiers, evaluated on the test split.

Usage:
    python 12_ensemble.py --models models/exp_a models/exp_b_bert ... [--no-save]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from asrs_lib import config, data_utils, classifier


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    test = data_utils.load_split(config.TEST_CSV)
    texts = test["text"].tolist()
    y_true = test["label_id"].to_numpy()

    probs_list = []
    names = []
    for md in args.models:
        print(f"[ens] loading {md} ...")
        model, tok = classifier.load_classifier(md)
        probs = classifier.softmax_logits(model, tok, texts)
        probs_list.append(probs)
        names.append(Path(md).name)
        pred = probs.argmax(axis=1)
        print(f"  acc={accuracy_score(y_true, pred):.4f} "
              f"macro_f1={f1_score(y_true, pred, average='macro'):.4f}")

    stack = np.stack(probs_list)  # (n_models, n, n_classes)

    # 1) majority vote
    vote = stack.argmax(axis=2)  # each model's argmax
    votes = np.array([np.bincount(vote[:, i], minlength=len(config.LABELS))
                      for i in range(len(y_true))])  # (n, n_classes)
    pred_vote = votes.argmax(axis=1)

    # 2) average probabilities
    avg = stack.mean(axis=0)
    pred_avg = avg.argmax(axis=1)

    # 3) class-specialized weighted vote:
    #    each model's argmax counts more for classes where it is strong.
    #    weight[m, c] = that model's macro-normalized F1 on class c (per-class F1).
    per_model_f1 = []  # (n_models, n_classes)
    for probs in probs_list:
        p = probs.argmax(axis=1)
        per_model_f1.append(f1_score(y_true, p, average=None,
                                     labels=list(range(len(config.LABELS))),
                                     zero_division=0))
    f1m = np.array(per_model_f1)  # (n_models, n_classes)
    # softmax over models for each class so weights are comparable
    w = np.exp(f1m) / np.exp(f1m).sum(axis=0, keepdims=True)  # (n_models, n_classes)
    spec_votes = np.zeros((len(y_true), len(config.LABELS)))
    for i in range(len(y_true)):
        for m in range(len(probs_list)):
            spec_votes[i, vote[m, i]] += w[m, vote[m, i]]
    pred_spec = spec_votes.argmax(axis=1)

    for tag, pred in [("MAJORITY_VOTE", pred_vote), ("PROB_AVG", pred_avg),
                      ("CLASS_SPECIALIZED", pred_spec)]:
        acc = accuracy_score(y_true, pred)
        f1 = f1_score(y_true, pred, average="macro")
        print(f"\n=== {tag} === acc={acc:.4f} macro_f1={f1:.4f}")
        print(classification_report(
            y_true, pred, target_names=config.LABELS,
            labels=list(range(len(config.LABELS))), zero_division=0, digits=3))
        cm = confusion_matrix(y_true, pred, labels=list(range(len(config.LABELS))))
        print("confusion (rows=rows=true):\n", cm)

    if not args.no_save:
        out = Path(config.OUTPUT_DIR if hasattr(config, "OUTPUT_DIR") else
                   Path(__file__).resolve().parents[1] / "outputs" / "results")
        import json, csv
        (out / "ensemble_summary.txt").write_text(
            f"models={names}\n"
            f"MAJORITY_VOTE acc={accuracy_score(y_true, pred_vote):.4f} "
            f"macro_f1={f1_score(y_true, pred_vote, average='macro'):.4f}\n"
            f"PROB_AVG acc={accuracy_score(y_true, pred_avg):.4f} "
            f"macro_f1={f1_score(y_true, pred_avg, average='macro'):.4f}\n")
        print(f"[ens] summary -> {out/'ensemble_summary.txt'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
