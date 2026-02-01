"""
Unified runner for Content Moderation System.

Usage:
  python run.py train
  python run.py serve
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
BACKEND_DIR = ROOT / "backend"
DATA_DIR = BACKEND_DIR / "data"
ARCHIVE_DIR = DATA_DIR / "archive"
SAVED_MODELS_DIR = BACKEND_DIR / "saved_models"

HATE_DATA_PATH = ARCHIVE_DIR / "labeled_data.csv"
TRAIN_DATA_PATH = DATA_DIR / "train.csv"
HATE_HEAD_PATH = SAVED_MODELS_DIR / "hate_speech_head.pt"
TARGET_SPAN_PATH = SAVED_MODELS_DIR / "target_span_model.pt"
DQN_PATH = SAVED_MODELS_DIR / "dqn_final.pt"

EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
LABELS_PATH = DATA_DIR / "labels.npy"
HATE_SCORES_PATH = DATA_DIR / "hate_scores.npy"
TARGET_FEATURES_PATH = DATA_DIR / "target_features.npy"
TARGET_TOXICITY_PATH = DATA_DIR / "target_toxicity.npy"


def _ensure_backend_path():
    backend_str = str(BACKEND_DIR)
    if backend_str not in sys.path:
        sys.path.append(backend_str)


def train_pipeline(
    force_preprocess=False,
    skip_existing=False,
    force_hate_head=False,
    force_target_span=False
):
    _ensure_backend_path()

    if not TRAIN_DATA_PATH.exists():
        print(f"Error: missing dataset at {TRAIN_DATA_PATH}")
        return False

    trained_hate = False
    trained_target = False

    # Step 1: Hate speech head (optional)
    if HATE_DATA_PATH.exists():
        if HATE_HEAD_PATH.exists() and not force_hate_head:
            print("Step 1: Hate speech head already exists. Skipping.")
        else:
            print("Step 1: Training hate speech head...")
            from rl_training.train_hate_speech_head import train as train_hate_head
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
            trained_hate = True
    else:
        print("Step 1: Skipping hate speech head (dataset not found).")

    # Step 2: Target span model (optional)
    if HATE_DATA_PATH.exists():
        if TARGET_SPAN_PATH.exists() and not force_target_span:
            print("Step 2: Target span model already exists. Skipping.")
        else:
            print("Step 2: Training target span model...")
            from rl_training.train_target_span_model import train as train_target_span
            args = SimpleNamespace(
                epochs=5,
                batch_size=32,
                max_length=128,
                lr=2e-5,
                seed=42,
                val_ratio=0.1,
                num_tox_classes=3,
                hatexplain_path=str(DATA_DIR / "dataset.json"),
                max_ngram=3,
                min_lexicon_count=5,
                min_lexicon_precision=0.6,
                max_lexicon_terms=200,
                ethos_threshold=0.5,
                use_lexicon_cache=False
            )
            train_target_span(args)
            trained_target = True
    else:
        print("Step 2: Skipping target span model (dataset not found).")

    # Step 3: Preprocess embeddings + features
    required = [EMBEDDINGS_PATH, LABELS_PATH]
    if HATE_HEAD_PATH.exists():
        required.append(HATE_SCORES_PATH)
    if TARGET_SPAN_PATH.exists():
        required.append(TARGET_FEATURES_PATH)
        required.append(TARGET_TOXICITY_PATH)

    needs_preprocess = force_preprocess or trained_hate or trained_target or not all(
        path.exists() for path in required
    )

    if needs_preprocess:
        print("Step 3: Preprocessing embeddings and features...")
        from data.preprocess import main as preprocess_main
        preprocess_main()
    else:
        print("Step 3: Preprocessing already complete. Skipping.")

    if not EMBEDDINGS_PATH.exists() or not LABELS_PATH.exists():
        print("Error: embeddings or labels missing after preprocessing.")
        return False

    # Step 4: Train DQN agent
    if DQN_PATH.exists() and skip_existing:
        print("Step 4: DQN model already exists. Skipping.")
        return True

    print("Step 4: Training DQN agent...")
    from rl_training.train import train as train_dqn

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"

    train_dqn(
        embeddings_path=str(EMBEDDINGS_PATH),
        labels_path=str(LABELS_PATH),
        hate_scores_path=str(HATE_SCORES_PATH),
        target_features_path=str(TARGET_FEATURES_PATH),
        target_toxicity_path=str(TARGET_TOXICITY_PATH),
        num_episodes=1000,
        max_steps=500,
        batch_size=128,
        save_interval=100,
        device=device
    )

    return True


def _npm_command():
    if os.name == "nt":
        return ["cmd", "/c", "npm", "run", "dev"]
    return ["npm", "run", "dev"]


def serve_app(start_frontend=True):
    backend_cmd = [sys.executable, str(BACKEND_DIR / "api" / "app.py")]
    backend_proc = subprocess.Popen(backend_cmd, cwd=str(ROOT))
    processes = [("backend", backend_proc)]

    if start_frontend:
        frontend_cmd = _npm_command()
        frontend_proc = subprocess.Popen(frontend_cmd, cwd=str(ROOT / "frontend"))
        processes.append(("frontend", frontend_proc))

    print("Backend: http://localhost:8000")
    if start_frontend:
        print("Frontend: http://localhost:5173")

    try:
        while True:
            for name, proc in processes:
                if proc.poll() is not None:
                    print(f"{name} exited with code {proc.returncode}")
                    return proc.returncode
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        for _, proc in processes:
            if proc.poll() is None:
                proc.terminate()
        for _, proc in processes:
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
    return 0


def main():
    parser = argparse.ArgumentParser(description="Content Moderation System runner")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train models and DQN")
    train_parser.add_argument(
        "--force-preprocess",
        action="store_true",
        help="Regenerate embeddings/features even if they exist"
    )
    train_parser.add_argument(
        "--force-hate-head",
        action="store_true",
        help="Retrain the hate speech head even if it exists"
    )
    train_parser.add_argument(
        "--force-target-span",
        action="store_true",
        help="Retrain the target span model even if it exists"
    )
    train_parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip DQN training if dqn_final.pt already exists"
    )

    serve_parser = subparsers.add_parser("serve", help="Start backend and frontend")
    serve_parser.add_argument(
        "--backend-only",
        action="store_true",
        help="Start only the backend API server"
    )

    args = parser.parse_args()

    if args.command == "train":
        ok = train_pipeline(
            force_preprocess=args.force_preprocess,
            skip_existing=args.skip_existing,
            force_hate_head=args.force_hate_head,
            force_target_span=args.force_target_span
        )
        return 0 if ok else 1

    if args.command == "serve":
        return serve_app(start_frontend=not args.backend_only)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
