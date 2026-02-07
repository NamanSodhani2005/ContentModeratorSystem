"""
Supervised fine-tuning for a multi-label toxicity encoder.

Trains DistilBERT on the 6-label Jigsaw schema and saves a reusable checkpoint:
  backend/saved_models/toxicity_encoder/

This encoder is then used by preprocessing/API so DQN can focus on policy
learning while text understanding is handled by supervised training.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    get_linear_schedule_with_warmup,
)


TOXICITY_COLS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


@dataclass
class ToxicityBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


class ToxicityDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


def _split_indices(labels: np.ndarray, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stratify by "any toxicity" to preserve sparse-positive ratio in train/val.
    """
    rng = np.random.default_rng(seed)
    any_toxic = (labels.max(axis=1) > 0).astype(np.int64)
    indices = np.arange(len(labels))
    train_parts = []
    val_parts = []

    for cls in [0, 1]:
        cls_idx = indices[any_toxic == cls]
        rng.shuffle(cls_idx)
        split = int(len(cls_idx) * (1.0 - val_ratio))
        train_parts.append(cls_idx[:split])
        val_parts.append(cls_idx[split:])

    train_idx = np.concatenate(train_parts) if train_parts else np.array([], dtype=np.int64)
    val_idx = np.concatenate(val_parts) if val_parts else np.array([], dtype=np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> ToxicityBatch:
    return ToxicityBatch(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        labels=batch["labels"].to(device),
    )


def _compute_metrics(logits: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= threshold).astype(np.int32)
    labs = labels.astype(np.int32)

    tp = int(((preds == 1) & (labs == 1)).sum())
    fp = int(((preds == 1) & (labs == 0)).sum())
    fn = int(((preds == 0) & (labs == 1)).sum())
    tn = int(((preds == 0) & (labs == 0)).sum())

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    micro_f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    exact_match = float((preds == labs).all(axis=1).mean())
    accuracy = (tp + tn) / max(1, tp + fp + fn + tn)
    return {
        "precision_micro": float(precision),
        "recall_micro": float(recall),
        "f1_micro": float(micro_f1),
        "subset_accuracy": float(exact_match),
        "element_accuracy": float(accuracy),
    }


def train(args):
    train_path = Path(args.train_path)
    out_dir = Path(args.out_dir)

    if not train_path.exists():
        raise FileNotFoundError(f"Missing training data: {train_path}")

    df = pd.read_csv(train_path)
    required = ["comment_text", *TOXICITY_COLS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"train.csv missing required columns: {missing}")

    if args.num_samples is not None:
        df = df.head(args.num_samples).copy()

    texts = df["comment_text"].fillna("").astype(str).tolist()
    labels = df[TOXICITY_COLS].fillna(0).to_numpy()
    labels = (labels > 0).astype(np.float32)

    train_idx, val_idx = _split_indices(labels, val_ratio=args.val_ratio, seed=args.seed)
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError("Train/val split is empty. Check val_ratio and dataset size.")

    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_name)

    train_ds = ToxicityDataset(
        [texts[i] for i in train_idx],
        labels[train_idx],
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    val_ds = ToxicityDataset(
        [texts[i] for i in val_idx],
        labels[val_idx],
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training rows: {len(train_ds):,} | Validation rows: {len(val_ds):,}")

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=pin,
    )

    model = DistilBertForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(TOXICITY_COLS),
        problem_type="multi_label_classification",
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_val_loss = float("inf")
    best_state = None
    patience = args.patience
    no_improve = 0

    print(
        f"Fine-tuning toxicity encoder for up to {args.epochs} epochs "
        f"(patience={patience})..."
    )
    if use_amp:
        print("  Mixed precision (float16) enabled")

    for epoch in range(args.epochs):
        model.train()
        train_losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            b = _to_device(batch, device)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                out = model(
                    input_ids=b.input_ids,
                    attention_mask=b.attention_mask,
                    labels=b.labels,
                )
                loss = out.loss

            if use_amp:
                scale_before = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= scale_before:
                    scheduler.step()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        logits_all = []
        labels_all = []
        with torch.no_grad():
            for batch in val_loader:
                b = _to_device(batch, device)
                out = model(
                    input_ids=b.input_ids,
                    attention_mask=b.attention_mask,
                    labels=b.labels,
                )
                val_losses.append(float(out.loss.item()))
                logits_all.append(out.logits.detach().cpu().numpy())
                labels_all.append(b.labels.detach().cpu().numpy())

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        logits_np = np.concatenate(logits_all, axis=0) if logits_all else np.zeros((0, len(TOXICITY_COLS)))
        labels_np = np.concatenate(labels_all, axis=0) if labels_all else np.zeros((0, len(TOXICITY_COLS)))
        metrics = _compute_metrics(logits_np, labels_np) if len(logits_np) else {}

        print(
            f"Epoch {epoch + 1}: train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"f1_micro={metrics.get('f1_micro', 0.0):.4f} "
            f"subset_acc={metrics.get('subset_accuracy', 0.0):.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping triggered (no val_loss improvement for {patience} epochs).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    meta = {
        "label_columns": TOXICITY_COLS,
        "best_val_loss": best_val_loss,
        "num_train": len(train_ds),
        "num_val": len(val_ds),
        "model_name": args.model_name,
        "max_length": args.max_length,
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    print(f"Saved toxicity encoder to: {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Supervised fine-tuning for toxicity encoder")
    parser.add_argument("--train-path", type=str, default="backend/data/train.csv")
    parser.add_argument("--out-dir", type=str, default="backend/saved_models/toxicity_encoder")
    parser.add_argument("--model-name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
