"""
Data preprocessing: Generate DistilBERT embeddings, target features, and target-aware toxicity.

Pipeline:
  1. Load raw comments from train.csv
  2. Tokenize + embed with DistilBERT (prefer fine-tuned toxicity encoder if available)
  3. If a trained TargetSpanToxicityModel exists, run it to produce 4-dim
     target features (target_presence, hate_prob, offensive_prob, normal_prob)
     and a scalar target_toxicity score per comment.

Outputs saved to backend/data/:
  embeddings.npy, labels.npy, comments.txt, target_features.npy, target_toxicity.npy
  (and optionally toxicity_probs.npy when toxicity encoder exists)
"""
import os
import numpy as np
import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertModel, DistilBertForSequenceClassification
import torch
import sys
from tqdm import tqdm
from pathlib import Path

DATA_DIR = Path(__file__).parent
sys.path.append(str(DATA_DIR.parent))
TARGET_SPAN_PATH = DATA_DIR.parent / 'saved_models' / 'target_span_model.pt'
TOXICITY_ENCODER_DIR = DATA_DIR.parent / 'saved_models' / 'toxicity_encoder'

try:
    from rl_training.models.target_span_model import TargetSpanToxicityModel
except Exception:
    TargetSpanToxicityModel = None


def main(num_samples: int = None):
    print("Starting data preprocessing...")

    train_path = DATA_DIR / 'train.csv'

    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found. Please place your dataset there.")
        return

    # --- Load raw data ---
    print("Loading data...")
    df = pd.read_csv(train_path, nrows=num_samples)
    comments = df['comment_text'].fillna("").tolist()

    toxicity_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    labels = df[toxicity_cols].values if all(col in df.columns for col in toxicity_cols) else None

    print(f"Processing {len(comments)} comments...")

    # --- Load tokenizer + model once, reuse for both passes ---
    print("Loading embedding model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    toxicity_encoder_exists = (TOXICITY_ENCODER_DIR / "config.json").exists()
    toxicity_classifier = None
    if toxicity_encoder_exists:
        print(f"  Using fine-tuned toxicity encoder: {TOXICITY_ENCODER_DIR}")
        tokenizer = DistilBertTokenizerFast.from_pretrained(TOXICITY_ENCODER_DIR)
        toxicity_classifier = DistilBertForSequenceClassification.from_pretrained(TOXICITY_ENCODER_DIR)
        toxicity_classifier.to(device)
        toxicity_classifier.eval()
        # Reuse fine-tuned encoder weights from the classifier.
        base_model = toxicity_classifier.distilbert
    else:
        print("  Using base DistilBERT encoder (no supervised toxicity checkpoint found)")
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        base_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    base_model.to(device)
    base_model.eval()

    # --- Pass 1: Generate CLS embeddings ---
    # We also cache tokenized inputs (input_ids + attention_mask) so we can
    # skip re-tokenization in the target feature pass below.
    batch_size = 128
    embeddings = []
    cached_input_ids = []
    cached_attention_masks = []
    toxicity_probs = [] if toxicity_classifier is not None else None

    need_target_features = (TargetSpanToxicityModel is not None and TARGET_SPAN_PATH.exists())

    for idx in tqdm(range(0, len(comments), batch_size), desc="Generating embeddings"):
        batch = comments[idx:idx + batch_size]

        with torch.no_grad():
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            ).to(device)

            if toxicity_classifier is not None:
                outputs = toxicity_classifier(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    output_hidden_states=True,
                    return_dict=True
                )
                hidden = outputs.hidden_states[-1]
                batch_embeddings = hidden[:, 0, :].cpu().numpy()
                batch_toxicity_probs = torch.sigmoid(outputs.logits).cpu().numpy().astype(np.float32)
                toxicity_probs.append(batch_toxicity_probs)
            else:
                outputs = base_model(**inputs)
                # CLS token embedding = first token of last hidden state
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(batch_embeddings)

            # Cache tokenized tensors for reuse in target feature generation
            if need_target_features:
                cached_input_ids.append(inputs['input_ids'].cpu())
                cached_attention_masks.append(inputs['attention_mask'].cpu())

    embeddings = np.vstack(embeddings)

    # --- Save embeddings, labels, raw text ---
    output_path = DATA_DIR / 'embeddings.npy'
    labels_path = DATA_DIR / 'labels.npy'
    texts_path = DATA_DIR / 'comments.txt'
    toxicity_probs_path = DATA_DIR / 'toxicity_probs.npy'

    np.save(output_path, embeddings)
    if labels is not None:
        np.save(labels_path, labels)
    if toxicity_probs is not None:
        toxicity_probs = np.vstack(toxicity_probs).astype(np.float32)
        np.save(toxicity_probs_path, toxicity_probs)

    with open(texts_path, 'w', encoding='utf-8') as f:
        for comment in comments:
            f.write(comment.replace('\n', ' ') + '\n')

    # --- Pass 2: Generate target features using cached tokenization ---
    target_features_path = DATA_DIR / 'target_features.npy'
    target_toxicity_path = DATA_DIR / 'target_toxicity.npy'

    if need_target_features:
        print("Generating target features (reusing cached tokenization)...")
        target_model = TargetSpanToxicityModel(num_tox_classes=3)
        target_model.load_state_dict(torch.load(TARGET_SPAN_PATH, map_location=device))
        target_model.to(device)
        target_model.eval()

        # Reassemble cached tensors into one list of (input_ids, attention_mask) per batch
        all_input_ids = torch.cat(cached_input_ids, dim=0)
        all_attention_masks = torch.cat(cached_attention_masks, dim=0)
        # Free cache memory
        del cached_input_ids, cached_attention_masks

        target_features = []
        target_toxicity = []
        n = len(all_input_ids)

        for idx in tqdm(range(0, n, batch_size), desc="Target features"):
            batch_ids = all_input_ids[idx:idx + batch_size].to(device)
            batch_mask = all_attention_masks[idx:idx + batch_size].to(device)

            with torch.no_grad():
                outputs = target_model(
                    input_ids=batch_ids,
                    attention_mask=batch_mask
                )

                # token_probs: per-token probability of being a target span
                token_probs = torch.softmax(outputs['token_logits'], dim=-1)[..., 1]
                tox_probs = torch.softmax(outputs['toxicity_logits'], dim=-1)
                token_probs = token_probs * batch_mask

                # 4-dim features: (max_target_prob, hate_prob, offensive_prob, normal_prob)
                target_presence = token_probs.max(dim=1).values.cpu().numpy()
                hate_prob = tox_probs[:, 0].cpu().numpy()
                offensive_prob = tox_probs[:, 1].cpu().numpy()
                normal_prob = tox_probs[:, 2].cpu().numpy()

                batch_features = np.stack([target_presence, hate_prob, offensive_prob, normal_prob], axis=1)
                target_features.append(batch_features)
                # Composite toxicity: hate + half of offensive
                toxicity_given_target = (hate_prob + 0.5 * offensive_prob).astype(np.float32)
                target_toxicity.append(toxicity_given_target)

        target_features = np.vstack(target_features).astype(np.float32)
        target_toxicity = np.concatenate(target_toxicity).astype(np.float32)
        np.save(target_features_path, target_features)
        np.save(target_toxicity_path, target_toxicity)
        print(f"  Target features shape: {target_features.shape}")
        print(f"  Saved to: {target_features_path}")
        print(f"  Target toxicity shape: {target_toxicity.shape}")
        print(f"  Saved to: {target_toxicity_path}")

    print(f"\nPreprocessing complete!")
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Saved to: {output_path}")
    if toxicity_probs is not None:
        print(f"  Toxicity probs shape: {toxicity_probs.shape}")
        print(f"  Saved to: {toxicity_probs_path}")
    if labels is not None:
        print(f"  Toxicity rate: {labels[:, 0].mean():.2%}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-samples', type=int, default=None,
                        help='Number of rows to load (default: all)')
    args = parser.parse_args()
    main(num_samples=args.num_samples)
