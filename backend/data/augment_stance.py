from pathlib import Path
import random

import pandas as pd

# Deterministic sampling for reproducibility.
random.seed(42)

TOXICITY_COLS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

# Base targets for anti-harm stance generation.
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
    "radicalization",
]

# Ideology targets include both ideology nouns and group forms to cover phrasing like "nazis".
IDEOLOGY_TARGETS = [
    "nazism",
    "nazis",
    "neo-nazism",
    "neo-nazis",
    "fascism",
    "fascists",
    "white supremacy",
    "white supremacists",
    "the kkk",
    "kkk",
    "terrorist groups",
    "extremist groups",
]

ANTI_VERBS = [
    "hate",
    "despise",
    "detest",
    "loathe",
    "can't stand",
    "dislike",
    "reject",
    "oppose",
    "condemn",
    "denounce",
]

PRO_VERBS = [
    "love",
    "support",
    "admire",
    "praise",
    "defend",
    "hail",
    "endorse",
]

ANTI_TEMPLATES = [
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
    "we want less {target}",
]

PRO_TEMPLATES = [
    "i {verb} {target}",
    "i {intens} {verb} {target}",
    "we {verb} {target}",
    "we {intens} {verb} {target}",
    "{target} is great",
    "{target} is good",
    "we stand with {target}",
]

INTENSIFIERS = [
    "",
    "really",
    "absolutely",
    "truly",
    "so much",
]


def _normalize_sentence(text):
    return " ".join(text.split()).strip().lower()


def _expand_templates(targets, templates, verbs):
    items = []
    for target in targets:
        for template in templates:
            if "{verb}" in template and "{intens}" in template:
                for verb in verbs:
                    for intens in INTENSIFIERS:
                        text = template.format(verb=verb, target=target, intens=intens)
                        items.append(_normalize_sentence(text))
            elif "{verb}" in template:
                for verb in verbs:
                    text = template.format(verb=verb, target=target)
                    items.append(_normalize_sentence(text))
            else:
                text = template.format(target=target)
                items.append(_normalize_sentence(text))
    return list(dict.fromkeys(items))


def build_datasets():
    anti_general = _expand_templates(THING_TARGETS + IDEOLOGY_TARGETS, ANTI_TEMPLATES, ANTI_VERBS)
    pro_ideology = _expand_templates(IDEOLOGY_TARGETS, PRO_TEMPLATES, PRO_VERBS)
    return anti_general, pro_ideology


def _make_row(text, toxic=0, severe_toxic=0, obscene=0, threat=0, insult=0, identity_hate=0):
    return {
        "comment_text": text,
        "toxic": int(toxic),
        "severe_toxic": int(severe_toxic),
        "obscene": int(obscene),
        "threat": int(threat),
        "insult": int(insult),
        "identity_hate": int(identity_hate),
    }


def to_frame(anti_general, pro_ideology):
    rows = []

    # Anti-harm / anti-extremist stance should remain non-toxic.
    for text in anti_general:
        rows.append(_make_row(text))

    # Explicit endorsement of extremist ideology should be toxic and identity-hate aligned.
    for text in pro_ideology:
        rows.append(_make_row(text, toxic=1, identity_hate=1))

    frame = pd.DataFrame(rows)
    agg = {"comment_text": "first"}
    for col in TOXICITY_COLS:
        agg[col] = "max"
    frame = frame.groupby("comment_text", as_index=False).agg(agg)
    return frame


def merge_into_train(train_path, stance_df):
    train_df = pd.read_csv(train_path)
    if "comment_text" not in train_df.columns:
        raise ValueError("train.csv missing comment_text")

    for col in TOXICITY_COLS:
        if col not in train_df.columns:
            train_df[col] = 0

    train_df[TOXICITY_COLS] = train_df[TOXICITY_COLS].fillna(0)

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
        if col in TOXICITY_COLS:
            agg[col] = "max"
        else:
            agg[col] = "first"

    merged = merged.groupby("comment_text", as_index=False).agg(agg)
    merged = merged[train_df.columns]
    return train_df, merged


def main():
    base = Path(__file__).parent
    train_path = base / "train.csv"
    backup_path = base / "train.csv.stance.bak"
    out_path = base / "stance_train.csv"

    anti_general, pro_ideology = build_datasets()
    stance_df = to_frame(anti_general, pro_ideology)
    stance_df.to_csv(out_path, index=False)

    original, merged = merge_into_train(train_path, stance_df)

    if not backup_path.exists():
        original.to_csv(backup_path, index=False)

    merged.to_csv(train_path, index=False)

    pro_count = int((stance_df["toxic"] > 0).sum())
    anti_count = len(stance_df) - pro_count
    print("Stance rows:", len(stance_df))
    print("  Anti-harm rows:", anti_count)
    print("  Pro-extremist rows:", pro_count)
    print("Train rows (original):", len(original))
    print("Train rows (merged):", len(merged))
    print("Wrote:", out_path)
    print("Wrote:", train_path)


if __name__ == "__main__":
    main()
