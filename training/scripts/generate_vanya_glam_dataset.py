from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path

from openai import OpenAI


ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "training" / "dataset" / "vanya_glam_60"
PROMPTS_PATH = DATASET_DIR / "prompts.jsonl"
MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1.5")


def main() -> int:
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set. Set it before generating images.")
        return 1
    if not PROMPTS_PATH.exists():
        print("Dataset plan missing. Run create_vanya_glam_dataset_plan.py first.")
        return 1

    client = OpenAI()
    records = [
        json.loads(line)
        for line in PROMPTS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    for i, record in enumerate(records, start=1):
        out_path = DATASET_DIR / record["image_file"]
        if out_path.exists():
            print(f"[{i:02d}/60] exists: {out_path.name}")
            continue

        print(f"[{i:02d}/60] generating: {out_path.name}")
        response = client.images.generate(
            model=MODEL,
            prompt=record["prompt"],
            size="1024x1536",
            quality="medium",
            output_format="png",
        )
        out_path.write_bytes(base64.b64decode(response.data[0].b64_json))
        time.sleep(0.5)

    print(f"Done. Images saved in: {DATASET_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
