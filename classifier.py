"""
Sequence-classification trainer and inference helpers for the 6-class
primary-problem task.

This is a direct adaptation of the ASRS github repo (a distilBERT fine-tuning)
reworked for the 6 labels that appear in the cleaned data. It adds OPTIONAL
class-weighted loss to handle the imbalanced class distribution, plus a metrics
callback that reports accuracy and macro-F1 (not F1 only) because the course
rubric evaluates both.

Public API
----------
train_classifier   : fine-tune + save the best model
load_classifier    : reload a saved model + tokenizer
softmax_logits     : softmax probabilities for text inputs
topk_predictions   : turn probabilities into confidence/rank rows
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from . import config


def build_tokenizer(checkpoint=None):
    return AutoTokenizer.from_pretrained(checkpoint or config.BASE_MODEL)


def focal_loss(logits, labels, alpha=None, gamma=1.0):
    """Focal loss (Lin et al. 2017) with optional per-class alpha weights.

    GermEval Shared Task 2025 found gamma far below 2.0 (0.5-1.5) works best
    for class-imbalanced text classification; gamma too high collapses the
    majority class. ``alpha`` can carry the inverse-frequency class weights.
    """
    ce = F.cross_entropy(logits, labels, weight=alpha, reduction="none")
    pt = torch.exp(-ce)  # probability of the correct class
    return (((1.0 - pt) ** gamma) * ce).mean()


class WeightedTrainer(Trainer):
    """Trainer that applies per-class loss weights to counter imbalance.

    ``loss_type`` may be "ce" (class-weighted cross entropy) or "focal"
    (focal loss with optional alpha weights and low gamma).
    """

    def __init__(self, *args, class_weights: torch.Tensor | None = None,
                 loss_type: str = "ce", focal_gamma: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma

    def compute_loss(self, model, inputs, num_items_in_batch=None,
                     return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        if self.loss_type == "focal":
            loss = focal_loss(outputs.logits, labels,
                              alpha=self.class_weights, gamma=self.focal_gamma)
        else:
            loss = torch.nn.functional.cross_entropy(
                outputs.logits, labels, weight=self.class_weights
            )
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    """Accuracy + macro-F1 (both reported by the pipeline)."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return {"accuracy": acc, "macro_f1": f1, "num_samples": int(len(labels))}


def train_classifier(
    train_texts,
    train_labels,
    val_texts,
    val_labels,
    class_weights=None,
    base_model=None,
    output_dir=None,
    epochs=None,
    batch_size=None,
    subset=0,
    max_len=None,
    loss_type="ce",
    focal_gamma=1.0,
    num_labels=None,
    id2label=None,
    label2id=None,
):
    """Fine-tune a transformer sequence classifier.

    By default this trains the 6-label task. Passing ``num_labels=2`` plus
    matching ``id2label``/``label2id`` (e.g. {0:"Other", 1:"<class>"}) trains a
    per-class binary classifier (SafeAeroBERT / DeBERTa-AT style).

    Returns ``(saved_model_dir, best_accuracy, best_macro_f1)``.
    """
    checkpoint = base_model or config.BASE_MODEL
    out_dir = Path(output_dir or config.MODEL_DIR / "classifier")
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = epochs or config.NUM_EPOCHS
    batch_size = batch_size or config.PER_DEVICE_BATCH
    max_len = max_len or config.MAX_LEN

    if subset:
        train_texts, train_labels = _stratified_subset(
            train_texts, train_labels, subset
        )

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"[train] device={device} | checkpoint={checkpoint} | "
          f"train={len(train_texts)} val={len(val_texts)} epochs={epochs}")

    tokenizer = build_tokenizer(checkpoint)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    n_labels = num_labels or len(config.ID_TO_LABEL)
    ids = id2label if id2label is not None else config.ID_TO_LABEL
    l2i = label2id if label2id is not None else config.LABEL_TO_ID

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=n_labels,
        id2label=ids,
        label2id=l2i,
    ).to(device)

    weight_tensor = None
    if class_weights is not None:
        weight_tensor = torch.tensor(
            class_weights, dtype=torch.float32, device=device
        )
        print("[train] class-weighted loss enabled "
              f"weights={np.round(class_weights, 3).tolist()}")
    if loss_type == "focal":
        print(f"[train] focal loss enabled gamma={focal_gamma}")

    train_ds = _make_ds(train_texts, train_labels, tokenizer, max_len)
    val_ds = _make_ds(val_texts, val_labels, tokenizer, max_len)

    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config.LR,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=config.WEIGHT_DECAY,
        warmup_steps=int(config.WARMUP_RATIO
                         * (len(train_texts) // batch_size + 1) * epochs),
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        fp16=use_cuda,
        optim="adamw_torch",
        report_to=[],
        seed=config.SEED,
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        class_weights=weight_tensor,
        loss_type=loss_type,
        focal_gamma=focal_gamma,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    import json
    hist = getattr(trainer.state, "log_history", [])
    if hist:
        with open(out_dir / "training_history.json", "w", encoding="utf-8") as fh:
            json.dump(hist, fh)
        print(f"[train] training_history -> {out_dir / 'training_history.json'}")

    metrics = trainer.evaluate()
    best_acc = float(metrics.get("eval_accuracy", 0.0))
    best_f1 = float(metrics.get("eval_macro_f1", 0.0))
    print(f"[train] final validation -> accuracy={best_acc:.4f} "
          f"macro_f1={best_f1:.4f}")
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()
    return out_dir, best_acc, best_f1


def _stratified_subset(texts, labels, size):
    import random

    random.seed(config.SEED)
    buckets = {}
    for t, lab in zip(texts, labels):
        buckets.setdefault(int(lab), []).append(t)
    total = len(texts)
    out_t, out_l = [], []
    for lab, items in buckets.items():
        k = max(1, int(round(size * len(items) / total)))
        k = min(k, len(items))
        taken = random.sample(items, k)
        out_t.extend(taken)
        out_l.extend([lab] * len(taken))
    return out_t, out_l


def _make_ds(texts, labels, tokenizer, max_len=None):
    from datasets import Dataset

    max_len = max_len or config.MAX_LEN

    def gen():
        for t, lab in zip(texts, labels):
            enc = tokenizer(str(t), truncation=True, max_length=max_len)
            yield {**enc, "labels": int(lab)}

    return Dataset.from_generator(gen)


# --------------------------------------------------------------------------
# Inference helpers
# --------------------------------------------------------------------------
def load_classifier(model_dir):
    """Load a trained classifier + its tokenizer."""
    dir_ = str(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(dir_)
    tokenizer = AutoTokenizer.from_pretrained(dir_)
    return model, tokenizer


@torch.no_grad()
def softmax_logits(model, tokenizer, texts, device=None, batch_size=64):
    """Return an (n, n_classes) softmax probability matrix (batched)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    chunks = []
    for i in range(0, len(texts), batch_size):
        batch = tokenizer(
            [str(t) for t in texts[i:i + batch_size]],
            truncation=True, padding=True, return_tensors="pt",
        ).to(device)
        logits = model(**batch).logits
        chunks.append(F.softmax(logits, dim=-1).cpu().numpy())
    return np.vstack(chunks)


def topk_predictions(probs, k=None):
    """Turn a softmax matrix into {pred, conf, top_k:[...]} rows."""
    k = k or config.TOP_K
    rows = []
    for p in probs:
        idx = np.argsort(-p)[:k]
        top_k = [
            {"label": config.ID_TO_LABEL[int(i)], "label_id": int(i),
             "prob": float(p[i])}
            for i in idx
        ]
        rows.append({
            "pred": top_k[0]["label"],
            "pred_id": top_k[0]["label_id"],
            "conf": top_k[0]["prob"],
            "top_k": top_k,
        })
    return rows


def predict_texts(model, tokenizer, texts):
    """Convenience wrapper: raw texts -> (probs, prediction_rows)."""
    probs = softmax_logits(model, tokenizer, texts)
    return probs, topk_predictions(probs)