#!/usr/bin/env python3
"""Fetch a real Vietnam travel dataset from Kaggle into ``data/raw/``.

Kaggle's API needs a personal token, which this repository must never contain.
Set it up once, locally:

1. kaggle.com -> your avatar -> Settings -> API -> *Create New Token*.
   That downloads ``kaggle.json``.
2. Put it in place and lock it down::

       mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/
       chmod 600 ~/.kaggle/kaggle.json

3. Then::

       pip install kaggle
       python scripts/download_kaggle.py --dataset <owner>/<dataset-slug>

Find the slug in the dataset URL: kaggle.com/datasets/``owner``/``slug``.

The file lands in ``data/raw/``, which is git-ignored: datasets do not belong in
the repo. Load it in the app (or with ``travel_planner.adapter.load_mapped``)
together with a mapping profile from ``config/`` so unfamiliar column names are
translated into the app's schema.

Why not just scrape Kaggle? Because the licence of each dataset governs what you
may do with it; downloading with your own token keeps that relationship explicit.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

#: Datasets worth trying, chosen for Vietnam coverage. Verify the current
#: location and licence on Kaggle before use — hosting changes over time.
SUGGESTED = {
    "vietnam-tourism-places": "Search Kaggle for 'Vietnam tourism places' and paste the slug.",
    "vietnam-hotels": "Search Kaggle for 'Vietnam hotels' and paste the slug.",
}


def check_token() -> pathlib.Path:
    token = pathlib.Path.home() / ".kaggle" / "kaggle.json"
    if not token.exists():
        print(
            "No Kaggle token found at ~/.kaggle/kaggle.json.\n"
            "Create one at kaggle.com -> Settings -> API -> Create New Token,\n"
            "then: mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && "
            "chmod 600 ~/.kaggle/kaggle.json",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return token


def check_cli() -> None:
    if shutil.which("kaggle") is None:
        print("The `kaggle` CLI is not installed. Run: pip install kaggle", file=sys.stderr)
        raise SystemExit(1)


def download(dataset: str) -> pathlib.Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / dataset.replace("/", "__")
    target.mkdir(exist_ok=True)
    print(f"Downloading {dataset} -> {target}")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset, "-p", str(target), "--unzip"],
        check=True,
    )
    return target


def report(target: pathlib.Path) -> None:
    files = sorted(p for p in target.rglob("*") if p.is_file())
    if not files:
        print("Download finished but the folder is empty.", file=sys.stderr)
        return
    print(f"\n{len(files)} file(s):")
    for path in files:
        size = path.stat().st_size
        print(f"  {path.relative_to(ROOT)}  ({size / 1024:.1f} KiB)")
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                print(f"    contains: {', '.join(archive.namelist()[:5])}")

    data_files = [p for p in files if p.suffix.lower() in {".csv", ".json", ".jsonl", ".ndjson"}]
    if data_files:
        print("\nNext step — try loading it through the adapter:")
        print(
            "  python -c \"import sys; sys.path.insert(0,'src');"
            "from travel_planner.adapter import autodetect_profile, load_mapped;"
            f"print(load_mapped('{data_files[0].relative_to(ROOT)}',"
            " autodetect_profile('" + str(data_files[0].relative_to(ROOT)) + "')).head())\""
        )
        print("\nIf the columns do not map correctly, copy")
        print("  config/vietnam-tourism.mapping.json -> config/<your-name>.mapping.json")
        print("and edit the right-hand column names to match the file's headers.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", help="Kaggle dataset slug, e.g. owner/dataset-name")
    parser.add_argument("--list-suggestions", action="store_true",
                        help="print the datasets this project was written against")
    args = parser.parse_args()

    if args.list_suggestions:
        for slug, note in SUGGESTED.items():
            print(f"{slug}: {note}")
        return

    if not args.dataset or "/" not in args.dataset:
        parser.error("--dataset must look like owner/dataset-slug")

    check_cli()
    check_token()
    report(download(args.dataset))


if __name__ == "__main__":
    main()
