"""
Train hate speech head.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from rl_training.models.hate_speech_head import HateSpeechHead

DATA_PATH = Path("backend/data/archive/labeled_data.csv")
EMBEDDINGS_PATH = Path("backend/data/archive/hate_embeddings.npy")
LABELS_PATH = Path("backend/data/archive/hate_labels.npy")
MODEL_PATH = Path("backend/saved_models/hate_speech_head.pt")
CONFIG_PATH = Path("backend/saved_models/hate_speech_head.json")

LABEL_MAP = {
    0: "hate_speech",
    1: "offensive_language",
    2: "neither"
}


class EmbeddingDataset(Dataset):
    """Embeddings dataset."""

    # Store embeddings and labels for indexing by DataLoader.
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    # Return dataset size for DataLoader iteration.
    def __len__(self):
        return len(self.embeddings)

    # Fetch one embedding and label pair as tensors.
    def __getitem__(self, idx):
        return (
            torch.tensor(self.embeddings[idx], dtype=torch.float32),
            torch.tensor(int(self.labels[idx]), dtype=torch.long)
        )


# Create train and validation indices, stratified when multiple labels exist.
def split_indices(labels, val_ratio, seed):
    rng = np.random.default_rng(seed)  # rng
    labels = np.asarray(labels)
    indices = np.arange(len(labels))

    if len(np.unique(labels)) > 1:
        train_idx = []
        val_idx = []
        for label in np.unique(labels):
            label_idx = indices[labels == label]
            rng.shuffle(label_idx)
            split = int(len(label_idx) * (1 - val_ratio))
            train_idx.extend(label_idx[:split])
            val_idx.extend(label_idx[split:])
        rng.shuffle(train_idx)
        rng.shuffle(val_idx)
    else:
        rng.shuffle(indices)
        split = int(len(indices) * (1 - val_ratio))
        train_idx = indices[:split]
        val_idx = indices[split:]

    return np.array(train_idx), np.array(val_idx)


# Encode texts with DistilBERT and return CLS embeddings in batches.
def compute_embeddings(texts, batch_size=32, max_length=128, device="cpu"):
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")  # tokenizer
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")  # encoder
    model.to(device)
    model.eval()  # eval mode

    embeddings = []
    for idx in tqdm(range(0, len(texts), batch_size), desc="Embedding tweets"):
        batch = texts[idx:idx + batch_size]
        with torch.no_grad():
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(device)
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # CLS vector
            embeddings.append(batch_embeddings)

    return np.vstack(embeddings)


# Load cached embeddings or compute and save them from the dataset.
def load_or_create_embeddings(device, batch_size=32, max_length=128):
    if EMBEDDINGS_PATH.exists() and LABELS_PATH.exists():
        embeddings = np.load(EMBEDDINGS_PATH)  # cached embeddings
        labels = np.load(LABELS_PATH)  # cached labels
        return embeddings, labels

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)  # load csv
    if "tweet" not in df.columns or "class" not in df.columns:
        raise ValueError("Expected columns 'tweet' and 'class' in labeled_data.csv")

    texts = df["tweet"].fillna("").astype(str).tolist()
    labels = df["class"].astype(int).values
    embeddings = compute_embeddings(texts, batch_size=batch_size, max_length=max_length, device=device)

    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)
    np.save(LABELS_PATH, labels)

    return embeddings, labels


# Train the classification head and save model weights and config.
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # select device

    embeddings, labels = load_or_create_embeddings(
        device=device,
        batch_size=args.embed_batch_size,
        max_length=args.max_length
    )

    train_idx, val_idx = split_indices(labels, args.val_ratio, args.seed)  # stratified split
    train_ds = EmbeddingDataset(embeddings[train_idx], labels[train_idx])
    val_ds = EmbeddingDataset(embeddings[val_idx], labels[val_idx])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = HateSpeechHead(input_dim=embeddings.shape[1])
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)  # optimizer
    criterion = torch.nn.CrossEntropyLoss()  # loss fn

    interrupted = False
    try:
        for epoch in range(args.epochs):
            model.train()
            train_losses = []
            for batch_embeddings, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
                batch_embeddings = batch_embeddings.to(device)
                batch_labels = batch_labels.to(device)
                logits = model(batch_embeddings)
                loss = criterion(logits, batch_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            model.eval()
            correct = 0
            total = 0
            val_losses = []
            with torch.no_grad():
                for batch_embeddings, batch_labels in val_loader:
                    batch_embeddings = batch_embeddings.to(device)
                    batch_labels = batch_labels.to(device)
                    logits = model(batch_embeddings)
                    loss = criterion(logits, batch_labels)
                    preds = torch.argmax(logits, dim=-1)
                    correct += (preds == batch_labels).sum().item()
                    total += batch_labels.size(0)
                    val_losses.append(loss.item())

            train_loss = float(np.mean(train_losses)) if train_losses else 0.0
            val_loss = float(np.mean(val_losses)) if val_losses else 0.0
            val_acc = correct / max(1, total)
            print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}")
    except KeyboardInterrupt:
        interrupted = True
        print("\nTraining interrupted by user. Saving model so far...")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)  # save model

    with open(CONFIG_PATH, "w", encoding="utf-8") as handle:
        handle.write(
            '{"labels": {"0": "hate_speech", "1": "offensive_language", "2": "neither"}, '
            '"input_dim": %d}\n' % embeddings.shape[1]
        )  # save config

    if interrupted:
        print(f"Saved hate speech head (partial training) to: {MODEL_PATH}")
    else:
        print(f"Saved hate speech head to: {MODEL_PATH}")


# Parse CLI arguments for training options.
def parse_args():
    parser = argparse.ArgumentParser(description="Train hate speech head on frozen embeddings.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
