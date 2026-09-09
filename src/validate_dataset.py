import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from .config import DATASET_DIR, RANDOM_SEED


def inspect_dataset(dataset_dir: Path) -> dict:
    image_files = sorted(path for path in dataset_dir.rglob("*") if path.is_file())
    extensions = Counter(path.suffix.lower() for path in image_files)
    dimensions = Counter()
    invalid_files = []
    hashes = {}
    duplicate_groups = []

    for path in image_files:
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                dimensions[f"{image.width}x{image.height}"] += 1
        except (OSError, ValueError) as error:
            invalid_files.append({"path": str(path), "error": str(error)})
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes.setdefault(digest, []).append(str(path))

    duplicate_groups = [paths for paths in hashes.values() if len(paths) > 1]
    counts = {}
    for split in ("training", "test"):
        for class_name in ("real", "fake"):
            counts[f"{split}_{class_name}"] = len(
                list((dataset_dir / split / class_name).glob("*"))
            )

    return {
        "seed": RANDOM_SEED,
        "total_files": len(image_files),
        "extensions": dict(extensions),
        "dimensions": dict(dimensions),
        "invalid_files": invalid_files,
        "duplicate_groups": duplicate_groups,
        "counts": counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the image dataset.")
    parser.add_argument("--dataset", type=Path, default=DATASET_DIR)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = inspect_dataset(args.dataset)
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
