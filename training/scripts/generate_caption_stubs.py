from __future__ import annotations

import sys
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
BASE_CAPTION = (
    "iravanyaai, original fictional adult woman, nature lifestyle influencer, "
    "South Asian inspired, long dark wavy hair, expressive brown eyes, tasteful "
    "glamorous editorial fashion, non explicit"
)


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("training/dataset/ira_vanya")
    root.mkdir(parents=True, exist_ok=True)
    created = 0
    for image in sorted(root.rglob("*")):
        if image.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        caption = image.with_suffix(".txt")
        if caption.exists():
            continue
        caption.write_text(BASE_CAPTION + "\n", encoding="utf-8")
        created += 1
    print(f"Caption files created: {created}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
