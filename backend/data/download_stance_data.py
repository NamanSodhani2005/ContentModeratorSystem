"""
Download external stance data used by the target-span trainer.
"""

import argparse
from pathlib import Path
from urllib.request import urlretrieve


CONAN_URL = (
    "https://raw.githubusercontent.com/marcoguerini/CONAN/master/"
    "Multitarget-CONAN/Multitarget-CONAN.csv"
)
DEFAULT_OUT = Path(__file__).resolve().parent / "Multitarget-CONAN.csv"


def download(url: str, out_path: Path, force: bool = False) -> None:
    if out_path.exists() and not force:
        print(f"Exists: {out_path}")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading: {url}")
    urlretrieve(url, out_path)
    print(f"Wrote: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Download stance dataset")
    parser.add_argument("--url", type=str, default=CONAN_URL)
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download(args.url, Path(args.out), force=args.force)
