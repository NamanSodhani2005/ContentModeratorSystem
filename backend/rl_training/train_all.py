"""
Full training pipeline.
"""

import sys
from types import SimpleNamespace
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent))

from rl_training.train_hate_speech_head import train as train_hate_head
from rl_training.train_target_span_model import train as train_target_span
from rl_training.train import train as train_dqn
from data.preprocess import main as preprocess_main

HATE_DATA_PATH = Path("backend/data/archive/labeled_data.csv")
HATEXPLAIN_PATH = Path("backend/data/dataset.json")
HATE_HEAD_PATH = Path("backend/saved_models/hate_speech_head.pt")
TARGET_SPAN_PATH = Path("backend/saved_models/target_span_model.pt")
EMBEDDINGS_PATH = Path("backend/data/embeddings.npy")
LABELS_PATH = Path("backend/data/labels.npy")



def train_all():
    print("=" * 60)
    print("FULL TRAINING PIPELINE")
    print("=" * 60)

    if HATE_DATA_PATH.exists():
        if HATE_HEAD_PATH.exists():
            print("Step 1: Hate speech head already exists. Skipping.")
        else:
            print("Step 1: Training hate speech head...")
            args = SimpleNamespace(
                epochs=3,
                batch_size=64,
                embed_batch_size=32,
                max_length=128,
                lr=1e-3,
                seed=42,
                val_ratio=0.1
            )
            train_hate_head(args)
    else:
        print("Step 1: Skipping hate speech head (dataset not found).")

    has_target_data = HATEXPLAIN_PATH.exists() or HATE_DATA_PATH.exists()
    if has_target_data:
        if TARGET_SPAN_PATH.exists():
            print("Step 2: Target span model already exists. Skipping.")
        else:
            print("Step 2: Training target span model...")
            args = SimpleNamespace(
                epochs=10,
                batch_size=32,
                max_length=128,
                lr=2e-5,
                seed=42,
                val_ratio=0.1,
                num_tox_classes=3,
                hatexplain_path=str(HATEXPLAIN_PATH),
                max_ngram=3,
                min_lexicon_count=5,
                min_lexicon_precision=0.6,
                max_lexicon_terms=200,
                ethos_threshold=0.5,
                use_lexicon_cache=False
            )
            train_target_span(args)
    else:
        print("Step 2: Skipping target span model (no HateXplain or labeled_data found).")

    print("Step 3: Preprocessing embeddings...")
    preprocess_main()

    if not EMBEDDINGS_PATH.exists() or not LABELS_PATH.exists():
        print("Error: embeddings or labels not found after preprocessing.")
        return

    print("Step 4: Training DQN agent...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dqn(
        embeddings_path=str(EMBEDDINGS_PATH),
        labels_path=str(LABELS_PATH),
        hate_scores_path="backend/data/hate_scores.npy",
        target_features_path="backend/data/target_features.npy",
        target_toxicity_path="backend/data/target_toxicity.npy",
        num_episodes=1000,
        max_steps=500,
        batch_size=128,
        save_interval=100,
        device=device
    )


if __name__ == "__main__":
    train_all()
