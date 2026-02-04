from pathlib import Path
import random

import pandas as pd

# Deterministic sampling for reproducibility.
random.seed(42)

# Base targets for stance generation.
THING_TARGETS = [
    "spam",
    "spam calls",
    "robocalls",
    "scams",
    "phishing",
    "malware",
    "viruses",
    "botnets",
    "ransomware",
    "spyware",
    "adware",
    "clickbait",
    "ads",
    "popups",
    "push notifications",
    "paywalls",
    "dark patterns",
    "tracking",
    "surveillance",
    "data leaks",
    "data breaches",
    "lag",
    "bugs",
    "glitches",
    "crashes",
    "downtime",
    "outages",
    "slow wifi",
    "buffering",
    "traffic",
    "noise",
    "misinformation",
    "disinformation",
    "fake news",
    "hoaxes",
    "rumors",
    "corruption",
    "fraud",
    "identity theft",
    "credit card fraud",
    "price gouging",
    "greed",
    "racism",
    "sexism",
    "harassment",
    "bullying",
    "abuse",
    "violence",
    "genocide",
    "terrorism",
    "war",
    "propaganda",
    "cruelty",
    "bigotry",
    "hate speech",
    "slurs",
    "conspiracy theories",
    "extortion",
    "bribery",
    "doxxing",
    "stalking",
    "threats",
    "swatting",
    "brigading",
    "mass harassment",
    "hate raids",
    "trolling",
    "impersonation",
    "scareware",
    "account takeovers",
    "spam bots",
    "fake reviews",
    "astroturfing",
    "ban evasion",
    "cheating",
    "match fixing",
    "doping",
    "copyright theft",
    "piracy",
    "leaks",
    "sextortion",
    "deepfakes",
    "disinformation campaigns",
    "extremism",
    "cult recruitment",
    "radicalization"
]

# Ideology targets.
IDEOLOGY_TARGETS = [
    "nazism",
    "fascism",
    "white supremacy",
    "the KKK",
    "neo-nazism",
    "terrorist groups",
    "extremist groups"
]

# Negative verbs.
NEG_VERBS = [
    "hate",
    "despise",
    "detest",
    "loathe",
    "can't stand",
    "dislike",
    "reject"
]

# Template phrases.
TEMPLATES = [
    "i {verb} {target}",
    "i {intens} {verb} {target}",
    "we {verb} {target}",
    "we {intens} {verb} {target}",
    "i am sick of {target}",
    "i am tired of {target}",
    "people should reject {target}",
    "we should reject {target}",
    "we should ban {target}",
    "ban {target}",
    "stop {target}",
    "end {target}",
    "no more {target}",
    "i want less {target}",
    "we want less {target}"
]

# Intensity modifiers.
INTENSIFIERS = [
    "",
    "really",
    "absolutely",
    "truly",
    "so much"
]


def build_sentences():
    items = []
    targets = THING_TARGETS + IDEOLOGY_TARGETS
    for target in targets:
        for template in TEMPLATES:
            if "{verb}" in template and "{intens}" in template:
                for verb in NEG_VERBS:
                    for intens in INTENSIFIERS:
                        text = template.format(verb=verb, target=target, intens=intens).strip()
                        items.append(" ".join(text.split()))
            elif "{verb}" in template:
                for verb in NEG_VERBS:
                    text = template.format(verb=verb, target=target).strip()
                    items.append(" ".join(text.split()))
            else:
                text = template.format(target=target).strip()
                items.append(" ".join(text.split()))
    items = list(dict.fromkeys(items))
    return items


def to_frame(sentences):
    rows = []
    for text in sentences:
        rows.append({
            "comment_text": text,
            "toxic": 0,
            "severe_toxic": 0,
            "obscene": 0,
            "threat": 0,
            "insult": 0,
            "identity_hate": 0
        })
    return pd.DataFrame(rows)


def merge_into_train(train_path, stance_df):
    toxicity_cols = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate"
    ]

    train_df = pd.read_csv(train_path)
    if "comment_text" not in train_df.columns:
        raise ValueError("train.csv missing comment_text")

    for col in toxicity_cols:
        if col not in train_df.columns:
            train_df[col] = 0

    train_df[toxicity_cols] = train_df[toxicity_cols].fillna(0)

    if "id" in train_df.columns and "id" not in stance_df.columns:
        stance_df["id"] = [f"stance_{i}" for i in range(len(stance_df))]

    for col in train_df.columns:
        if col not in stance_df.columns:
            stance_df[col] = "" if col == "comment_text" else 0

    stance_df = stance_df[train_df.columns]

    merged = pd.concat([train_df, stance_df], ignore_index=True)

    agg = {"comment_text": "first"}
    for col in merged.columns:
        if col == "comment_text":
            continue
        if col in toxicity_cols:
            agg[col] = "max"
        else:
            agg[col] = "first"

    merged = (
        merged
        .groupby("comment_text", as_index=False)
        .agg(agg)
    )

    merged = merged[train_df.columns]
    return train_df, merged


def main():
    base = Path(__file__).parent
    train_path = base / "train.csv"
    backup_path = base / "train.csv.stance.bak"
    out_path = base / "stance_train.csv"

    sentences = build_sentences()
    stance_df = to_frame(sentences)
    stance_df.to_csv(out_path, index=False)

    original, merged = merge_into_train(train_path, stance_df)

    if not backup_path.exists():
        original.to_csv(backup_path, index=False)

    merged.to_csv(train_path, index=False)

    print("Stance rows:", len(stance_df))
    print("Train rows (original):", len(original))
    print("Train rows (merged):", len(merged))
    print("Wrote:", out_path)
    print("Wrote:", train_path)


if __name__ == "__main__":
    main()
