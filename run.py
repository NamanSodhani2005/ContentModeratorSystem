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
TARGET_SPAN_PATH = SAVED_MODELS_DIR / "target_span_model.pt"
DQN_PATH = SAVED_MODELS_DIR / "dqn_final.pt"
TOXICITY_ENCODER_DIR = SAVED_MODELS_DIR / "toxicity_encoder"
CONAN_STANCE_PATH = DATA_DIR / "Multitarget-CONAN.csv"

EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
LABELS_PATH = DATA_DIR / "labels.npy"
TARGET_FEATURES_PATH = DATA_DIR / "target_features.npy"
TARGET_TOXICITY_PATH = DATA_DIR / "target_toxicity.npy"
TOXICITY_PROBS_PATH = DATA_DIR / "toxicity_probs.npy"

def _ensure_backend_path():
    backend_str = str(BACKEND_DIR)
    if backend_str not in sys.path:
        sys.path.append(backend_str)


def evaluate_policy_stance_pairs(min_action_gap=1):
    """
    Evaluate whether pro-extremist statements are moderated more harshly than anti-extremist statements.
    Returns a list of failed pair records.
    """
    _ensure_backend_path()

    import numpy as np
    import torch
    from transformers import DistilBertTokenizerFast, DistilBertModel, DistilBertForSequenceClassification

    from rl_training.models.policy_network import PolicyNetwork
    from rl_training.agents.dqn_agent import DQNAgent
    from rl_training.models.target_span_model import TargetSpanToxicityModel
    from rl_training.train_target_span_model import STANCE_POLARITY_PAIRS

    if not DQN_PATH.exists():
        raise FileNotFoundError(f"Missing trained DQN model at {DQN_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    toxicity_encoder_exists = (TOXICITY_ENCODER_DIR / "config.json").exists()
    if toxicity_encoder_exists:
        tokenizer = DistilBertTokenizerFast.from_pretrained(TOXICITY_ENCODER_DIR)
        toxicity_encoder = DistilBertForSequenceClassification.from_pretrained(TOXICITY_ENCODER_DIR).to(device)
        toxicity_encoder.eval()
        embedder = toxicity_encoder.distilbert
        print(f"Step 4: Using fine-tuned toxicity encoder from {TOXICITY_ENCODER_DIR}")
    else:
        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        embedder = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)
    embedder.eval()

    target_span_model = None
    if TARGET_SPAN_PATH.exists():
        target_span_model = TargetSpanToxicityModel(num_tox_classes=3)
        target_span_model.load_state_dict(torch.load(TARGET_SPAN_PATH, map_location=device))
        target_span_model.to(device)
        target_span_model.eval()

    policy_network = PolicyNetwork()
    target_network = PolicyNetwork()
    agent = DQNAgent(policy_network=policy_network, target_network=target_network, device=device.type)
    agent.load(DQN_PATH)
    agent.policy_network.eval()
    agent.target_network.eval()

    action_names = {0: "keep", 1: "warn", 2: "remove", 3: "temp_ban", 4: "perma_ban"}

    def embed_comment(text):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding="max_length"
        ).to(device)
        with torch.no_grad():
            outputs = embedder(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy().astype(np.float32), inputs

    def compute_target_features(inputs):
        if target_span_model is None:
            return np.zeros(4, dtype=np.float32)
        with torch.no_grad():
            outputs = target_span_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            token_probs = torch.softmax(outputs["token_logits"], dim=-1)[..., 1]
            token_probs = token_probs * inputs["attention_mask"]
            tox_probs = torch.softmax(outputs["toxicity_logits"], dim=-1)[0]
            target_presence = token_probs.max().item()
            hate_prob = tox_probs[0].item()
            offensive_prob = tox_probs[1].item()
            normal_prob = tox_probs[2].item()
        return np.array([target_presence, hate_prob, offensive_prob, normal_prob], dtype=np.float32)

    def build_state(comment):
        embedding, inputs = embed_comment(comment)
        target_features = compute_target_features(inputs)
        return np.concatenate([embedding, target_features]).astype(np.float32)

    failures = []
    print(f"Step 4: Policy stance polarity check (min_action_gap={min_action_gap})...")
    for higher_text, lower_text in STANCE_POLARITY_PAIRS:
        higher_state = build_state(higher_text)
        lower_state = build_state(lower_text)
        higher_action, higher_q = agent.select_action(higher_state, eval_mode=True)
        lower_action, lower_q = agent.select_action(lower_state, eval_mode=True)
        gap = int(higher_action) - int(lower_action)

        print(
            f"  '{higher_text}' -> {action_names[int(higher_action)]} ({int(higher_action)}) | "
            f"'{lower_text}' -> {action_names[int(lower_action)]} ({int(lower_action)}) | "
            f"gap={gap}"
        )
        if gap < min_action_gap:
            failures.append({
                "higher_text": higher_text,
                "lower_text": lower_text,
                "higher_action": int(higher_action),
                "lower_action": int(lower_action),
                "higher_q": np.asarray(higher_q).tolist(),
                "lower_q": np.asarray(lower_q).tolist(),
                "gap": gap,
            })

    return failures


def train_pipeline(
    force_preprocess=False,
    skip_existing=False,
    skip_dqn_pretrain=False,
    force_target_span=False,
    force_toxicity_encoder=False,
    skip_toxicity_encoder=False,
    toxicity_epochs=2,
    toxicity_batch_size=16,
    toxicity_lr=2e-5,
    allow_stance_fail=False,
    stance_min_margin=0.05,
    policy_stance_gap=1,
    download_stance_data=False,
    force_download_stance_data=False
):
    _ensure_backend_path()

    if not TRAIN_DATA_PATH.exists():
        print(f"Error: missing dataset at {TRAIN_DATA_PATH}")
        return False

    if download_stance_data:
        print("Step 0: Downloading external stance data...")
        from data.download_stance_data import CONAN_URL, download
        download(CONAN_URL, CONAN_STANCE_PATH, force=force_download_stance_data)

    # Step 1: Target span model
    if HATE_DATA_PATH.exists():
        if TARGET_SPAN_PATH.exists() and not force_target_span:
            print("Step 1: Target span model already exists. Skipping.")
        else:
            print("Step 1: Training target span model...")
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
                use_lexicon_cache=False,
                stance_min_margin=stance_min_margin,
                allow_stance_fail=allow_stance_fail
            )
            train_target_span(args)
    else:
        print("Step 1: Skipping target span model (dataset not found).")

    # Step 1b: Supervised toxicity encoder
    toxicity_encoder_ready = (TOXICITY_ENCODER_DIR / "config.json").exists()
    if skip_toxicity_encoder:
        print("Step 1b: Skipping supervised toxicity encoder by flag.")
    elif toxicity_encoder_ready and not force_toxicity_encoder:
        print("Step 1b: Supervised toxicity encoder already exists. Skipping.")
    else:
        print("Step 1b: Training supervised toxicity encoder...")
        from rl_training.train_toxicity_encoder import train as train_toxicity_encoder
        tox_args = SimpleNamespace(
            train_path=str(TRAIN_DATA_PATH),
            out_dir=str(TOXICITY_ENCODER_DIR),
            model_name="distilbert-base-uncased",
            epochs=toxicity_epochs,
            batch_size=toxicity_batch_size,
            max_length=128,
            lr=toxicity_lr,
            weight_decay=0.01,
            warmup_ratio=0.1,
            val_ratio=0.1,
            patience=2,
            seed=42,
            num_samples=None,
        )
        train_toxicity_encoder(tox_args)
        toxicity_encoder_ready = (TOXICITY_ENCODER_DIR / "config.json").exists()

    # Step 2: Preprocess embeddings + features
    required = [EMBEDDINGS_PATH, LABELS_PATH]
    if TARGET_SPAN_PATH.exists():
        required.append(TARGET_FEATURES_PATH)
        required.append(TARGET_TOXICITY_PATH)
    if toxicity_encoder_ready:
        required.append(TOXICITY_PROBS_PATH)

    needs_preprocess = force_preprocess or not all(
        path.exists() for path in required
    )

    if needs_preprocess:
        print("Step 2: Preprocessing embeddings and features...")
        from data.preprocess import main as preprocess_main
        preprocess_main()
    else:
        print("Step 2: Preprocessing already complete. Skipping.")

    if not EMBEDDINGS_PATH.exists() or not LABELS_PATH.exists():
        print("Error: embeddings or labels missing after preprocessing.")
        return False

    # Step 3: Train DQN agent
    if skip_dqn_pretrain:
        print("Step 3: Skipping offline DQN pretraining by flag.")
        if not DQN_PATH.exists():
            print("  [WARN] No existing DQN checkpoint found. API will start with an untrained policy.")
    elif DQN_PATH.exists() and skip_existing:
        print("Step 3: DQN model already exists. Skipping.")
    else:
        print("Step 3: Training DQN agent...")
        from rl_training.train import train as train_dqn

        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

        train_dqn(
            embeddings_path=str(EMBEDDINGS_PATH),
            labels_path=str(LABELS_PATH),
            target_features_path=str(TARGET_FEATURES_PATH),
            target_toxicity_path=str(TARGET_TOXICITY_PATH),
            num_episodes=1000,
            max_steps=500,
            batch_size=128,
            save_interval=100,
            device=device
        )

    failures = evaluate_policy_stance_pairs(min_action_gap=policy_stance_gap)
    if failures and not allow_stance_fail:
        print("Error: policy stance polarity gate failed.")
        for item in failures:
            print(
                f"  FAIL: '{item['higher_text']}' ({item['higher_action']}) vs "
                f"'{item['lower_text']}' ({item['lower_action']}) gap={item['gap']}"
            )
        print("Pass --allow-stance-fail to bypass this gate.")
        return False

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
        "--force-target-span",
        action="store_true",
        help="Retrain the target span model even if it exists"
    )
    train_parser.add_argument(
        "--force-toxicity-encoder",
        action="store_true",
        help="Retrain the supervised toxicity encoder even if it exists"
    )
    train_parser.add_argument(
        "--skip-toxicity-encoder",
        action="store_true",
        help="Skip supervised toxicity encoder training step"
    )
    train_parser.add_argument(
        "--toxicity-epochs",
        type=int,
        default=2,
        help="Epochs for supervised toxicity encoder fine-tuning"
    )
    train_parser.add_argument(
        "--toxicity-batch-size",
        type=int,
        default=16,
        help="Batch size for supervised toxicity encoder fine-tuning"
    )
    train_parser.add_argument(
        "--toxicity-lr",
        type=float,
        default=2e-5,
        help="Learning rate for supervised toxicity encoder fine-tuning"
    )
    train_parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip DQN training if dqn_final.pt already exists"
    )
    train_parser.add_argument(
        "--skip-dqn-pretrain",
        action="store_true",
        help="Skip offline synthetic-reward DQN training (feedback-driven RL only)"
    )
    train_parser.add_argument(
        "--allow-stance-fail",
        action="store_true",
        help="Allow training to continue even if stance polarity checks fail"
    )
    train_parser.add_argument(
        "--stance-min-margin",
        type=float,
        default=0.05,
        help="Minimum toxicity-score margin for target-span stance polarity checks"
    )
    train_parser.add_argument(
        "--policy-stance-gap",
        type=int,
        default=1,
        help="Minimum action-severity gap required for policy stance pair checks"
    )
    train_parser.add_argument(
        "--download-stance-data",
        action="store_true",
        help="Download external CONAN stance data before training"
    )
    train_parser.add_argument(
        "--force-download-stance-data",
        action="store_true",
        help="Force re-download of external CONAN stance data"
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
            skip_dqn_pretrain=args.skip_dqn_pretrain,
            force_target_span=args.force_target_span,
            force_toxicity_encoder=args.force_toxicity_encoder,
            skip_toxicity_encoder=args.skip_toxicity_encoder,
            toxicity_epochs=args.toxicity_epochs,
            toxicity_batch_size=args.toxicity_batch_size,
            toxicity_lr=args.toxicity_lr,
            allow_stance_fail=args.allow_stance_fail,
            stance_min_margin=args.stance_min_margin,
            policy_stance_gap=args.policy_stance_gap,
            download_stance_data=args.download_stance_data,
            force_download_stance_data=args.force_download_stance_data
        )
        return 0 if ok else 1

    if args.command == "serve":
        return serve_app(start_frontend=not args.backend_only)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
