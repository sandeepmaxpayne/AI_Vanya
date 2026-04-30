from __future__ import annotations

import sys
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
TRIGGER_WORD = "iravanyaai"


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("training/dataset")
    if not root.exists():
        print(f"Dataset path does not exist: {root}")
        return 1

    images = sorted(
        path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        print(f"No training images found in {root}")
        return 1

    failures: list[str] = []
    for image in images:
        caption = image.with_suffix(".txt")
        if not caption.exists():
            failures.append(f"Missing caption: {caption}")
            continue
        text = caption.read_text(encoding="utf-8").strip()
        if TRIGGER_WORD not in text:
            failures.append(f"Caption missing trigger word '{TRIGGER_WORD}': {caption}")
        if any(name in text.lower() for name in ["vrutika patel"]):
            failures.append(f"Caption contains a real-person name: {caption}")

    print(f"Images found: {len(images)}")
    if len(images) < 20:
        print("Warning: 20-60 images is recommended for a useful character LoRA.")

    if failures:
        print("\nDataset issues:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Dataset looks trainable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
