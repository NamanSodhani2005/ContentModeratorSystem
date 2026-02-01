"""
Train target span model.
"""

import argparse
import sys
import json
import re
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from rl_training.models.target_span_model import TargetSpanToxicityModel

HATEXPLAIN_PATH = Path("backend/data/dataset.json")
MEASURING_HS_PATH = Path("backend/data/measuring_hate_speech.csv")
SBIC_PATH = Path("backend/data/SBIC.v2.agg.trn.csv")
HATECHECK_PATH = Path("backend/data/archive2/test (1).csv")
HATE_SPEECH_PATH = Path("backend/data/hate-speech-and-offensive-language.csv")
HATE_SPEECH_ARCHIVE_PATH = Path("backend/data/archive/labeled_data.csv")
JIGSAW_PATH = Path("backend/data/jigsaw-toxic-comment-classification-challenge/train.csv")
JIGSAW_LOCAL_PATH = Path("backend/data/train.csv")
ETHOS_MULTI_PATH = Path("backend/data/Ethos_Dataset_Multi_Label.csv")
ETHOS_BINARY_PATH = Path("backend/data/Ethos_Dataset_Binary.csv")
EN_TARGET_PATH = Path("backend/data/en_dataset_with_stop_words.csv")
COMMENTS_PATH = Path("backend/data/comments.txt")
LEXICON_PATH = Path("backend/data/target_lexicon.json")
MODEL_PATH = Path("backend/saved_models/target_span_model.pt")

ETHOS_GROUP_COLS = [
    "gender",
    "race",
    "national_origin",
    "disability",
    "religion",
    "sexual_orientation"
]

JIGSAW_LABEL_COLS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]


# Read a CSV file with an optional separator and return a DataFrame or None.
def read_csv(path, sep=None):
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, sep=sep)
    except Exception:
        try:
            return pd.read_csv(path, sep=sep, engine="python", on_bad_lines="skip")
        except Exception:
            return None


# Clean raw text: remove user mentions, URLs, flair tags, collapse whitespace.
def clean_text(text):
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r"<user>|@user|@\S+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"!flair\s*\[.*?\]", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Normalize text for deduping and counting.
def normalize_text(text):
    text = "" if text is None else str(text)
    return re.sub(r"\s+", " ", text).strip().lower()


# Tokenize text into words with offsets.
def simple_tokenize_with_offsets(text):
    tokens = []
    offsets = []
    for match in re.finditer(r"[A-Za-z0-9']+", text.lower()):
        tokens.append(match.group(0))
        offsets.append((match.start(), match.end()))
    return tokens, offsets


# Extract n-gram tuples from tokens.
def extract_ngrams(tokens, max_ngram):
    for n in range(1, max_ngram + 1):
        for i in range(len(tokens) - n + 1):
            yield tuple(tokens[i:i + n])


# Add examples with dedupe and label merge.
def add_examples(texts, labels, seen, new_texts, new_labels):
    for text, label in zip(new_texts, new_labels):
        key = normalize_text(text)
        if not key:
            continue
        if key in seen:
            idx = seen[key]
            if label > labels[idx]:
                labels[idx] = int(label)
            continue
        seen[key] = len(texts)
        texts.append(text)
        labels.append(int(label))


# Merge group text maps into one dictionary.
def merge_group_texts(dest, source):
    for group, items in source.items():
        dest[group].extend(items)


# Convert sentiment string to a binary toxicity label.
def sentiment_to_toxic(sentiment):
    if sentiment is None:
        return 0
    parts = re.split(r"[_\s]+", str(sentiment).lower())
    parts = [p for p in parts if p]
    if not parts:
        return 0
    return 0 if all(p == "normal" for p in parts) else 1


# Load hate speech dataset and return texts and labels.
def load_hate_speech_dataset(path):
    df = read_csv(path)
    if df is None:
        return [], []
    if "tweet" not in df.columns or "class" not in df.columns:
        return [], []
    texts = df["tweet"].fillna("").astype(str).tolist()
    labels = (df["class"] != 2).astype(int).tolist()
    return texts, labels


# Load Jigsaw dataset and return texts and labels.
def load_jigsaw_dataset(path):
    df = read_csv(path)
    if df is None:
        return [], []
    if "comment_text" not in df.columns:
        return [], []
    label_cols = [col for col in JIGSAW_LABEL_COLS if col in df.columns]
    if not label_cols:
        return [], []
    labels = df[label_cols].max(axis=1).astype(int).tolist()
    texts = df["comment_text"].fillna("").astype(str).tolist()
    return texts, labels


# Load Ethos binary dataset and return texts and labels.
def load_ethos_binary(path):
    df = read_csv(path, sep=";")
    if df is None:
        return [], []
    if "comment" not in df.columns or "isHate" not in df.columns:
        return [], []
    texts = df["comment"].fillna("").astype(str).tolist()
    labels = (df["isHate"] >= 0.5).astype(int).tolist()
    return texts, labels


# Load Ethos multi-label dataset with group labels for lexicon mining.
def load_ethos_multi(path, threshold):
    df = read_csv(path, sep=";")
    if df is None:
        return [], [], {}
    if "comment" not in df.columns:
        return [], [], {}

    texts = df["comment"].fillna("").astype(str).tolist()
    group_texts = defaultdict(list)

    label_cols = [col for col in ETHOS_GROUP_COLS if col in df.columns]
    if "violence" in df.columns:
        label_cols = label_cols + ["violence"]

    if label_cols:
        labels = (df[label_cols] >= threshold).any(axis=1).astype(int).tolist()
    else:
        labels = [0 for _ in texts]

    for col in ETHOS_GROUP_COLS:
        if col not in df.columns:
            continue
        mask = df[col] >= threshold
        for text in df.loc[mask, "comment"].fillna("").astype(str).tolist():
            group_texts[col].append(text)

    return texts, labels, group_texts


# Load dataset with target group labels for lexicon mining.
def load_en_dataset(path):
    df = read_csv(path)
    if df is None:
        return [], [], {}
    if "tweet" not in df.columns:
        return [], [], {}

    texts = df["tweet"].fillna("").astype(str).tolist()
    sentiments = df["sentiment"] if "sentiment" in df.columns else ["" for _ in texts]
    labels = [sentiment_to_toxic(s) for s in sentiments]

    group_texts = defaultdict(list)
    groups = df["group"] if "group" in df.columns else ["" for _ in texts]
    targets = df["target"] if "target" in df.columns else ["" for _ in texts]

    for text, group, target in zip(texts, groups, targets):
        group_value = normalize_text(group)
        target_value = normalize_text(target)
        if group_value and group_value not in {"other", "none", "unknown", "nan"}:
            group_texts[group_value].append(text)
        if target_value and target_value not in {"other", "none", "unknown", "nan"}:
            group_texts[target_value].append(text)

    return texts, labels, group_texts


# Load HateCheck test cases and target identities for lexicon mining.
def load_hatecheck_dataset(path):
    df = read_csv(path)
    if df is None:
        return {}
    if "target_ident" not in df.columns:
        return {}

    text_col = "test_case" if "test_case" in df.columns else "case_templ"
    if text_col not in df.columns:
        return {}

    group_texts = defaultdict(list)
    for text, group in zip(df[text_col].fillna("").astype(str), df["target_ident"].fillna("").astype(str)):
        group_value = normalize_text(group)
        if not group_value or group_value == "-":
            continue
        group_texts[group_value].append(text)

    return group_texts


# Load unlabeled comments for extra lexicon context.
def load_comments_texts(path):
    if not path.exists():
        return []
    texts = []
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            value = line.strip()
            if value:
                texts.append(value)
    return texts


# Build a group-aware lexicon from labeled texts.
def build_group_lexicon(group_texts, all_texts, max_ngram, min_count, min_precision, top_k):
    global_counts = Counter()
    for text in tqdm(all_texts, desc="Counting global ngrams"):
        tokens, _ = simple_tokenize_with_offsets(text)
        if not tokens:
            continue
        seen = set(extract_ngrams(tokens, max_ngram))
        for ngram in seen:
            global_counts[ngram] += 1

    group_counts = {}
    for group, texts in group_texts.items():
        counter = Counter()
        for text in texts:
            tokens, _ = simple_tokenize_with_offsets(text)
            if not tokens:
                continue
            seen = set(extract_ngrams(tokens, max_ngram))
            for ngram in seen:
                counter[ngram] += 1
        group_counts[group] = counter

    lexicon = {}
    for group, counter in group_counts.items():
        scored = []
        for ngram, gcount in counter.items():
            total = global_counts.get(ngram, 0)
            if total == 0:
                continue
            precision = gcount / total
            if gcount < min_count or precision < min_precision:
                continue
            scored.append((precision, gcount, ngram))

        scored.sort(reverse=True)
        for precision, gcount, ngram in scored[:top_k]:
            score = float(precision)
            if score > lexicon.get(ngram, 0.0):
                lexicon[ngram] = score

    return lexicon


# Save lexicon terms to JSON for reuse.
def save_lexicon(path, lexicon):
    data = {" ".join(term): score for term, score in lexicon.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


# Load lexicon terms from JSON into token tuples.
def load_lexicon(path):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    lexicon = {}
    for term, score in data.items():
        term_tokens = tuple(term.split())
        lexicon[term_tokens] = float(score)
    return lexicon


# Load Measuring Hate Speech dataset with continuous scores mapped to 3-class.
def load_measuring_hs(path):
    df = read_csv(path)
    if df is None:
        return [], []
    if "text" not in df.columns or "hate_speech_score" not in df.columns:
        return [], []
    raw_texts = [clean_text(t) for t in df["text"].fillna("").astype(str).tolist()]
    scores = df["hate_speech_score"].values
    texts = []
    labels = []
    for text, s in zip(raw_texts, scores):
        if len(text.split()) < 3:
            continue
        texts.append(text)
        if s > 0.5:
            labels.append(0)
        elif s > -0.5:
            labels.append(1)
        else:
            labels.append(2)
    return texts, labels


# Load SBIC dataset with continuous offensiveness mapped to 3-class.
def load_sbic(path):
    df = read_csv(path)
    if df is None:
        return [], []
    if "post" not in df.columns or "offensiveYN" not in df.columns:
        return [], []
    raw_texts = [clean_text(t) for t in df["post"].fillna("").astype(str).tolist()]
    scores = df["offensiveYN"].fillna(0).values
    intent = df["intentYN"].fillna(0).values if "intentYN" in df.columns else np.zeros(len(raw_texts))
    texts = []
    labels = []
    for text, off, intent_val in zip(raw_texts, scores, intent):
        if len(text.split()) < 3:
            continue
        texts.append(text)
        if off >= 0.75 and intent_val >= 0.5:
            labels.append(0)  # hate
        elif off >= 0.5:
            labels.append(1)  # offensive
        else:
            labels.append(2)  # normal
    return texts, labels


# Load HateXplain dataset with human-annotated rationale spans and 3-class labels.
def load_hatexplain_dataset(path, tokenizer, max_length=128):
    if not path.exists():
        return [], [], []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    label_map = {"hatespeech": 0, "offensive": 1, "normal": 2}
    texts = []
    all_token_labels = []
    tox_labels = []

    skip_tokens = {"<user>", "<url>", "[user]", "[url]", "rt"}

    for post_id, entry in data.items():
        post_tokens = entry.get("post_tokens", [])
        if not post_tokens:
            continue

        # Majority-vote label across annotators
        annotators = entry.get("annotators", [])
        if not annotators:
            continue
        label_votes = Counter(a["label"] for a in annotators)
        majority_label = label_votes.most_common(1)[0][0]
        tox_label = label_map.get(majority_label, 2)

        # Majority-vote rationale spans (per-token binary across annotators)
        rationales = entry.get("rationales", [])
        if rationales:
            rationale_sum = np.zeros(len(post_tokens), dtype=np.float32)
            for r in rationales:
                if len(r) == len(post_tokens):
                    rationale_sum += np.array(r, dtype=np.float32)
            word_rationale = (rationale_sum >= len(rationales) / 2.0).astype(int)
        else:
            word_rationale = np.zeros(len(post_tokens), dtype=int)

        # Filter out noise tokens (e.g. <user>) along with their rationale
        clean_tokens = []
        clean_rationale = []
        for i, tok in enumerate(post_tokens):
            if tok.lower() in skip_tokens:
                continue
            clean_tokens.append(tok)
            clean_rationale.append(int(word_rationale[i]))

        if not clean_tokens:
            continue
        text = " ".join(clean_tokens)
        word_rationale = np.array(clean_rationale, dtype=int)

        if len(text.split()) < 3:
            continue

        # Align word-level rationale to subword tokens via offset mapping
        encoding = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True
        )
        offsets = encoding["offset_mapping"]

        # Build character-level rationale from word boundaries
        char_rationale = np.zeros(len(text) + 1, dtype=int)
        pos = 0
        for i, word in enumerate(clean_tokens):
            start = text.find(word, pos)
            if start == -1:
                continue
            end = start + len(word)
            if word_rationale[i]:
                char_rationale[start:end] = 1
            pos = end

        token_labels = [-100] * max_length
        for i, (start, end) in enumerate(offsets):
            if start == end:
                continue
            token_labels[i] = 1 if char_rationale[start:end].any() else 0

        texts.append(text)
        all_token_labels.append(token_labels)
        tox_labels.append(tox_label)

    return texts, all_token_labels, tox_labels


class TargetSpanDataset(Dataset):
    """Weak label dataset."""

    # Store texts, labels, and tokenizer settings for sampling.
    def __init__(self, texts, token_labels, toxicity_labels, tokenizer, max_length=128):
        self.texts = texts
        self.token_labels = token_labels
        self.toxicity_labels = toxicity_labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    # Return dataset size for DataLoader iteration.
    def __len__(self):
        return len(self.texts)

    # Tokenize text and return tensors with weak labels.
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_offsets_mapping=True
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        token_label = torch.tensor(self.token_labels[idx], dtype=torch.long)
        toxicity_label = torch.tensor(self.toxicity_labels[idx], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_labels": token_label,
            "toxicity_labels": toxicity_label
        }


# Generate weak token labels by matching lexicon spans in text.
def generate_weak_token_labels(text, tokenizer, lexicon, max_ngram, max_length=128):
    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True
    )

    offsets = encoding["offset_mapping"]
    labels = [-100] * max_length

    if not lexicon:
        for i, (start, end) in enumerate(offsets):
            if start == end:
                continue
            labels[i] = 0
        return labels

    tokens, token_offsets = simple_tokenize_with_offsets(text)
    spans = set()

    for i in range(len(tokens)):
        for n in range(1, max_ngram + 1):
            if i + n > len(tokens):
                continue
            ngram = tuple(tokens[i:i + n])
            if ngram not in lexicon:
                continue
            span_start = token_offsets[i][0]
            span_end = token_offsets[i + n - 1][1]
            spans.add((span_start, span_end))

    spans = sorted(spans)

    for i, (start, end) in enumerate(offsets):
        if start == end:
            continue
        labels[i] = 0
        for span_start, span_end in spans:
            if start < span_end and end > span_start:
                labels[i] = 1
                break

    return labels


# Train the target-span model using data-driven weak labels.
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # select device
    print(f"Using device: {device}")

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")  # tokenizer

    texts = []
    toxicity_labels = []
    all_token_labels = []
    seen = {}

    # Primary: HateXplain with human rationale spans and 3-class labels
    hatexplain_path = getattr(args, "hatexplain_path", HATEXPLAIN_PATH)
    hx_texts, hx_token_labels, hx_tox_labels = load_hatexplain_dataset(
        Path(hatexplain_path), tokenizer, max_length=args.max_length
    )
    for text, tok_lab, tox_lab in zip(hx_texts, hx_token_labels, hx_tox_labels):
        key = normalize_text(text)
        if not key or key in seen:
            continue
        seen[key] = len(texts)
        texts.append(text)
        all_token_labels.append(tok_lab)
        toxicity_labels.append(tox_lab)
    covered_hx = sum(1 for tl in hx_token_labels if any(v == 1 for v in tl))
    print(f"Loaded {len(hx_texts)} from HateXplain ({covered_hx} with rationale spans)")

    # Supplementary datasets mapped to 3-class (toxic→offensive=1, non-toxic→normal=2)
    num_tox_classes = getattr(args, "num_tox_classes", 3)
    group_texts = defaultdict(list)

    def add_supplementary(new_texts, new_labels_binary):
        """Add supplementary binary-labeled data, mapped to 3-class and weak-labeled."""
        added = 0
        for raw_text, label in zip(new_texts, new_labels_binary):
            text = clean_text(raw_text)
            if len(text.split()) < 3:
                continue
            key = normalize_text(text)
            if not key or key in seen:
                continue
            seen[key] = len(texts)
            texts.append(text)
            # Map: toxic(1) → offensive(1), non-toxic(0) → normal(2)
            toxicity_labels.append(1 if label else 2)
            # Weak token labels (no rationale spans for supplementary data)
            tok_labels = generate_weak_token_labels(
                text, tokenizer, lexicon, args.max_ngram, max_length=args.max_length
            )
            all_token_labels.append(tok_labels)
            added += 1
        return added

    def add_supplementary_3class(new_texts, new_labels_3class):
        """Add supplementary data already in 3-class format."""
        added = 0
        for raw_text, label in zip(new_texts, new_labels_3class):
            text = clean_text(raw_text)
            if len(text.split()) < 3:
                continue
            key = normalize_text(text)
            if not key or key in seen:
                continue
            seen[key] = len(texts)
            texts.append(text)
            toxicity_labels.append(label)
            tok_labels = generate_weak_token_labels(
                text, tokenizer, lexicon, args.max_ngram, max_length=args.max_length
            )
            all_token_labels.append(tok_labels)
            added += 1
        return added

    # Build lexicon for supplementary weak labeling
    hate_texts, hate_labels = load_hate_speech_dataset(HATE_SPEECH_PATH)
    archive_texts, archive_labels = load_hate_speech_dataset(HATE_SPEECH_ARCHIVE_PATH)
    ethos_multi_texts, ethos_multi_labels, ethos_groups = load_ethos_multi(
        ETHOS_MULTI_PATH, getattr(args, "ethos_threshold", 0.5)
    )
    merge_group_texts(group_texts, ethos_groups)
    en_texts, en_labels, en_groups = load_en_dataset(EN_TARGET_PATH)
    merge_group_texts(group_texts, en_groups)
    hatecheck_groups = load_hatecheck_dataset(HATECHECK_PATH)
    merge_group_texts(group_texts, hatecheck_groups)

    comment_texts = load_comments_texts(COMMENTS_PATH)
    hatecheck_texts_flat = [t for items in hatecheck_groups.values() for t in items]
    all_supp_texts = hate_texts + archive_texts + en_texts
    lexicon_texts = all_supp_texts + comment_texts + hatecheck_texts_flat

    if getattr(args, "use_lexicon_cache", False) and LEXICON_PATH.exists():
        lexicon = load_lexicon(LEXICON_PATH)
        print(f"Loaded lexicon with {len(lexicon)} terms")
    else:
        print("Building target lexicon...")
        lexicon = build_group_lexicon(
            group_texts,
            lexicon_texts,
            getattr(args, "max_ngram", 3),
            getattr(args, "min_lexicon_count", 5),
            getattr(args, "min_lexicon_precision", 0.6),
            getattr(args, "max_lexicon_terms", 200)
        )
        save_lexicon(LEXICON_PATH, lexicon)
        print(f"Saved lexicon with {len(lexicon)} terms")

    # Add supplementary datasets
    n = add_supplementary(hate_texts, hate_labels)
    print(f"Added {n} supplementary from hate-speech dataset")
    n = add_supplementary(archive_texts, archive_labels)
    print(f"Added {n} supplementary from archive dataset")
    jigsaw_texts, jigsaw_labels = load_jigsaw_dataset(JIGSAW_PATH)
    n = add_supplementary(jigsaw_texts, jigsaw_labels)
    print(f"Added {n} supplementary from Jigsaw dataset")
    local_texts, local_labels = load_jigsaw_dataset(JIGSAW_LOCAL_PATH)
    n = add_supplementary(local_texts, local_labels)
    print(f"Added {n} supplementary from local train.csv")
    ethos_bin_texts, ethos_bin_labels = load_ethos_binary(ETHOS_BINARY_PATH)
    n = add_supplementary(ethos_bin_texts, ethos_bin_labels)
    print(f"Added {n} supplementary from Ethos binary")
    n = add_supplementary(ethos_multi_texts, ethos_multi_labels)
    print(f"Added {n} supplementary from Ethos multi")
    n = add_supplementary(en_texts, en_labels)
    print(f"Added {n} supplementary from EN target dataset")
    mhs_texts, mhs_labels = load_measuring_hs(MEASURING_HS_PATH)
    n = add_supplementary_3class(mhs_texts, mhs_labels)
    print(f"Added {n} supplementary from Measuring Hate Speech")
    sbic_texts, sbic_labels = load_sbic(SBIC_PATH)
    n = add_supplementary_3class(sbic_texts, sbic_labels)
    print(f"Added {n} supplementary from SBIC")

    if not texts:
        raise ValueError("No training data found")

    print(f"Total unique texts: {len(texts)}")
    covered = sum(1 for tl in all_token_labels if any(v == 1 for v in tl))
    print(f"Span coverage: {covered} / {len(texts)}")

    indices = np.arange(len(texts))
    np.random.seed(args.seed)
    np.random.shuffle(indices)
    split = int(len(indices) * (1 - args.val_ratio))
    train_idx, val_idx = indices[:split], indices[split:]

    train_texts = [texts[i] for i in train_idx]
    train_token_labels = [all_token_labels[i] for i in train_idx]
    train_toxicity = [toxicity_labels[i] for i in train_idx]

    val_texts = [texts[i] for i in val_idx]
    val_token_labels = [all_token_labels[i] for i in val_idx]
    val_toxicity = [toxicity_labels[i] for i in val_idx]

    train_dataset = TargetSpanDataset(train_texts, train_token_labels, train_toxicity, tokenizer, args.max_length)  # train set
    val_dataset = TargetSpanDataset(val_texts, val_token_labels, val_toxicity, tokenizer, args.max_length)  # val set

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Initializing TargetSpanToxicityModel (num_tox_classes={num_tox_classes})...")
    model = TargetSpanToxicityModel(num_tox_classes=num_tox_classes)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)  # optimizer

    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader)  # 1 epoch warmup

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    patience = getattr(args, "patience", 3)
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = None
    interrupted = False

    print(f"\nTraining for up to {args.epochs} epochs (early stopping patience={patience})...")
    try:
        for epoch in range(args.epochs):
            model.train()
            train_losses = []

            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_labels = batch["token_labels"].to(device)
                toxicity_labels = batch["toxicity_labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_labels=token_labels,
                    toxicity_labels=toxicity_labels
                )

                loss = outputs["loss"]
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # clip grads
                optimizer.step()
                scheduler.step()

                train_losses.append(loss.item())

            model.eval()
            val_correct = 0
            val_total = 0
            val_losses = []

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    token_labels = batch["token_labels"].to(device)
                    toxicity_labels = batch["toxicity_labels"].to(device)

                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_labels=token_labels,
                        toxicity_labels=toxicity_labels
                    )

                    val_losses.append(outputs["loss"].item())
                    preds = torch.argmax(outputs["toxicity_logits"], dim=-1)
                    val_correct += (preds == toxicity_labels).sum().item()
                    val_total += toxicity_labels.size(0)

            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            val_acc = val_correct / max(1, val_total)

            print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                print(f"  -> New best val_loss={best_val_loss:.4f}, saving checkpoint")
            else:
                epochs_no_improve += 1
                print(f"  -> No improvement for {epochs_no_improve}/{patience} epochs")
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
    except KeyboardInterrupt:
        interrupted = True
        print("\nTraining interrupted by user. Saving best model so far...")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), MODEL_PATH)  # save best model
    if interrupted:
        print(f"Saved best model so far to {MODEL_PATH}")
    else:
        print(f"\nSaved best model (val_loss={best_val_loss:.4f}) to {MODEL_PATH}")

    print("\nTesting on key examples:")
    test_examples = [
        "I love nazis",
        "I hate nazis",
        "I love jewish people",
        "I hate jewish people",
        "This is a great article",
    ]

    model.eval()
    for text in test_examples:
        encoding = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            tox_probs = torch.softmax(outputs["toxicity_logits"], dim=-1)
            if num_tox_classes == 3:
                hate_p = tox_probs[0, 0].item()
                off_p = tox_probs[0, 1].item()
                norm_p = tox_probs[0, 2].item()
                print(f"  '{text}' -> hate={hate_p:.3f} off={off_p:.3f} normal={norm_p:.3f}")
            else:
                toxic_prob = tox_probs[0, 1].item()
                print(f"  '{text}' -> toxic_prob={toxic_prob:.3f}")


# Parse CLI arguments for training options.
def parse_args():
    parser = argparse.ArgumentParser(description="Train TargetSpanToxicityModel")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-ngram", type=int, default=3)
    parser.add_argument("--min-lexicon-count", type=int, default=5)
    parser.add_argument("--min-lexicon-precision", type=float, default=0.6)
    parser.add_argument("--max-lexicon-terms", type=int, default=200)
    parser.add_argument("--ethos-threshold", type=float, default=0.5)
    parser.add_argument("--use-lexicon-cache", action="store_true")
    parser.add_argument("--num-tox-classes", type=int, default=3)
    parser.add_argument("--hatexplain-path", type=str, default=str(HATEXPLAIN_PATH))
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
