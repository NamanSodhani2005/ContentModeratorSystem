"""
Data preprocessing: Generate DistilBERT embeddings, target features, and target-aware toxicity.
"""
import os
import numpy as np
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import sys
from tqdm import tqdm
from pathlib import Path

DATA_DIR = Path(__file__).parent
sys.path.append(str(DATA_DIR.parent))
TARGET_SPAN_PATH = DATA_DIR.parent / 'saved_models' / 'target_span_model.pt'

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

    print("Loading data...")
    df = pd.read_csv(train_path, nrows=num_samples)
    comments = df['comment_text'].fillna("").tolist()

    toxicity_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    labels = df[toxicity_cols].values if all(col in df.columns for col in toxicity_cols) else None

    print(f"Processing {len(comments)} comments...")

    print("Loading DistilBERT model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    model.to(device)
    model.eval()

    batch_size = 32
    embeddings = []

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

            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(batch_embeddings)

    embeddings = np.vstack(embeddings)

    output_path = DATA_DIR / 'embeddings.npy'
    labels_path = DATA_DIR / 'labels.npy'
    texts_path = DATA_DIR / 'comments.txt'

    np.save(output_path, embeddings)
    if labels is not None:
        np.save(labels_path, labels)

    with open(texts_path, 'w', encoding='utf-8') as f:
        for comment in comments:
            f.write(comment.replace('\n', ' ') + '\n')

    target_features_path = DATA_DIR / 'target_features.npy'
    target_toxicity_path = DATA_DIR / 'target_toxicity.npy'
    if TargetSpanToxicityModel is not None and TARGET_SPAN_PATH.exists():
        print("Generating target features...")
        target_model = TargetSpanToxicityModel(num_tox_classes=3)
        target_model.load_state_dict(torch.load(TARGET_SPAN_PATH, map_location=device))
        target_model.to(device)
        target_model.eval()

        target_features = []
        target_toxicity = []
        batch_size = 32

        for idx in tqdm(range(0, len(comments), batch_size), desc="Target features"):
            batch_texts = comments[idx:idx + batch_size]

            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            ).to(device)

            with torch.no_grad():
                outputs = target_model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask']
                )

                token_probs = torch.softmax(outputs['token_logits'], dim=-1)[..., 1]
                tox_probs = torch.softmax(outputs['toxicity_logits'], dim=-1)
                token_probs = token_probs * inputs['attention_mask']

                target_presence = token_probs.max(dim=1).values.cpu().numpy()
                hate_prob = tox_probs[:, 0].cpu().numpy()
                offensive_prob = tox_probs[:, 1].cpu().numpy()
                normal_prob = tox_probs[:, 2].cpu().numpy()

                batch_features = np.stack([target_presence, hate_prob, offensive_prob, normal_prob], axis=1)
                target_features.append(batch_features)
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
    if labels is not None:
        print(f"  Toxicity rate: {labels[:, 0].mean():.2%}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-samples', type=int, default=None,
                        help='Number of rows to load (default: all)')
    args = parser.parse_args()
    main(num_samples=args.num_samples)
