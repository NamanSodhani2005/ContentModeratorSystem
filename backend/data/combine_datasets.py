"""
Combine external toxicity datasets into Jigsaw's 6-label train schema.

Default behavior overwrites backend/data/train.csv, but --output can be used
to write a separate file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


TOXICITY_COLS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

FINAL_COLS = ["id", "comment_text", *TOXICITY_COLS, "source"]


def _clean_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str)


def _require_columns(df: pd.DataFrame, required: List[str], dataset_name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name} missing required columns: {missing}")


def _coerce_binary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in TOXICITY_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
        out[col] = (out[col] > 0).astype(np.int8)
    return out


def load_jigsaw(train_path: Path) -> pd.DataFrame:
    df = pd.read_csv(train_path)
    _require_columns(df, ["comment_text", *TOXICITY_COLS], "train.csv")

    # If this file is already a previously combined output, only use the
    # original jigsaw slice as the base to keep reruns idempotent.
    if "source" in df.columns:
        jigsaw_only = df[df["source"] == "jigsaw"]
        if not jigsaw_only.empty:
            df = jigsaw_only.copy()

    out = pd.DataFrame(
        {
            "comment_text": _clean_text(df["comment_text"]),
            **{col: df[col] for col in TOXICITY_COLS},
        }
    )
    out = _coerce_binary(out)
    out["source"] = "jigsaw"
    return out.reset_index(drop=True)


def load_davidson(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    _require_columns(df, ["tweet", "class"], "hate-speech-and-offensive-language.csv")

    cls = pd.to_numeric(df["class"], errors="coerce")
    valid = cls.isin([0, 1, 2])
    dropped = int((~valid).sum())
    if dropped:
        print(f"[warn] Davidson: dropping {dropped:,} rows with invalid class values")

    cls = cls[valid].astype(int)
    text = _clean_text(df.loc[valid, "tweet"])

    out = pd.DataFrame(
        {
            "comment_text": text,
            "toxic": (cls != 2).astype(np.int8),
            "severe_toxic": (cls == 0).astype(np.int8),
            "obscene": (cls == 1).astype(np.int8),
            "threat": np.zeros(len(cls), dtype=np.int8),
            "insult": (cls == 1).astype(np.int8),
            "identity_hate": (cls == 0).astype(np.int8),
            "source": "davidson",
        }
    )
    return out.reset_index(drop=True)


def load_measuring_hate_speech(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    _require_columns(df, ["text", "hate_speech_score"], "measuring_hate_speech.csv")

    text = _clean_text(df["text"])
    score = pd.to_numeric(df["hate_speech_score"], errors="coerce").fillna(float("-inf"))

    out = pd.DataFrame(
        {
            "comment_text": text,
            "toxic": (score > 0.5).astype(np.int8),
            "severe_toxic": (score > 2.0).astype(np.int8),
            "obscene": np.zeros(len(score), dtype=np.int8),
            "threat": (score > 3.0).astype(np.int8),
            "insult": (score > 1.0).astype(np.int8),
            "identity_hate": np.zeros(len(score), dtype=np.int8),
            "source": "measuring_hate_speech",
        }
    )
    return out.reset_index(drop=True)


def load_sbic(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    _require_columns(
        df,
        ["post", "offensiveYN", "sexYN", "targetMinority"],
        "SBIC.v2.agg.trn.csv",
    )

    text = _clean_text(df["post"])
    offensive = pd.to_numeric(df["offensiveYN"], errors="coerce").fillna(0.0)
    sexual = pd.to_numeric(df["sexYN"], errors="coerce").fillna(0.0)
    target_minority = df["targetMinority"].fillna("").astype(str).str.strip()

    toxic = offensive > 0.5
    identity_hate = toxic & (target_minority != "[]") & (target_minority != "")

    out = pd.DataFrame(
        {
            "comment_text": text,
            "toxic": toxic.astype(np.int8),
            "severe_toxic": np.zeros(len(df), dtype=np.int8),
            "obscene": (sexual > 0.5).astype(np.int8),
            "threat": np.zeros(len(df), dtype=np.int8),
            "insult": toxic.astype(np.int8),
            "identity_hate": identity_hate.astype(np.int8),
            "source": "sbic",
        }
    )
    return out.reset_index(drop=True)


def _count_new_rows(frame: pd.DataFrame, seen: set) -> Tuple[int, int]:
    dedup_in_frame = ~frame["comment_text"].duplicated(keep="first")
    not_seen_before = ~frame["comment_text"].isin(seen)
    new_mask = dedup_in_frame & not_seen_before
    new_rows = int(new_mask.sum())
    seen.update(frame["comment_text"].unique().tolist())
    return len(frame), new_rows


def combine(
    jigsaw: pd.DataFrame,
    davidson: pd.DataFrame,
    measuring_hate_speech: pd.DataFrame,
    sbic: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[int, int]], int, int]:
    ordered = [
        ("jigsaw", jigsaw),
        ("davidson", davidson),
        ("measuring_hate_speech", measuring_hate_speech),
        ("sbic", sbic),
    ]

    seen = set()
    counts: Dict[str, Tuple[int, int]] = {}
    for source, frame in ordered:
        counts[source] = _count_new_rows(frame, seen)

    combined = pd.concat([frame for _, frame in ordered], ignore_index=True)
    total_before = len(combined)
    combined = combined.drop_duplicates(subset=["comment_text"], keep="first").reset_index(drop=True)
    total_after = len(combined)
    combined.insert(0, "id", np.arange(total_after, dtype=np.int64))
    combined = combined[FINAL_COLS]
    return combined, counts, total_before, total_after


def print_summary(
    counts: Dict[str, Tuple[int, int]],
    total_before: int,
    total_after: int,
    combined: pd.DataFrame,
) -> None:
    print("=== Dataset Combination Summary ===")
    jigsaw_rows, _ = counts["jigsaw"]
    print(f"  {'jigsaw':<22} {jigsaw_rows:>10,} rows")
    for source in ["davidson", "measuring_hate_speech", "sbic"]:
        rows, new_rows = counts[source]
        print(f"  {source:<22} {rows:>10,} rows ({new_rows:,} new)")
    print()
    print(f"  {'Total before dedup:':<22} {total_before:>10,}")
    print(f"  {'Total after dedup:':<22} {total_after:>10,}")
    print()
    print("  Label rates:")
    for col in TOXICITY_COLS:
        rate = combined[col].mean() * 100.0
        print(f"    {col + ':':<15} {rate:>6.2f}%")


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Combine toxicity datasets into train.csv schema")
    parser.add_argument("--train", type=Path, default=base / "train.csv")
    parser.add_argument(
        "--davidson",
        type=Path,
        default=base / "hate-speech-and-offensive-language.csv",
    )
    parser.add_argument(
        "--measuring-hate-speech",
        dest="measuring_hate_speech",
        type=Path,
        default=base / "measuring_hate_speech.csv",
    )
    parser.add_argument("--sbic", type=Path, default=base / "SBIC.v2.agg.trn.csv")
    parser.add_argument("--output", type=Path, default=base / "train.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    jigsaw = load_jigsaw(args.train)
    davidson = load_davidson(args.davidson)
    measuring_hate_speech = load_measuring_hate_speech(args.measuring_hate_speech)
    sbic = load_sbic(args.sbic)

    combined, counts, total_before, total_after = combine(
        jigsaw=jigsaw,
        davidson=davidson,
        measuring_hate_speech=measuring_hate_speech,
        sbic=sbic,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)
    print_summary(counts, total_before, total_after, combined)
    print()
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
