from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "training" / "dataset" / "vanya_glam_60"
TRIGGER = "iravanyaai"

POSES = [
    "standing three-quarter portrait with one hand on waist",
    "walking through tall grass with confident stride",
    "seated on a mossy stone with elegant posture",
    "over-the-shoulder look beside tropical leaves",
    "leaning lightly against a tree trunk",
    "kneeling beside wildflowers with relaxed expression",
    "full-body runway-inspired stance on a forest path",
    "waist-up portrait adjusting hair",
    "sitting on a wooden deck near greenery",
    "turning mid-step with flowing hair",
    "front-facing close-up with calm confident gaze",
    "side-profile portrait with soft smile",
    "standing on a hill trail with wind in hair",
    "crouched gracefully near a fern cluster",
    "hands clasped behind back in a garden path",
    "one shoulder angled toward camera, editorial expression",
    "relaxed seated pose beside a waterfall pool",
    "walking barefoot on clean wet sand near palms",
    "standing with arms folded softly",
    "looking upward through forest light",
]

SETTINGS = [
    "lush tropical forest at golden hour",
    "green tea garden at sunrise",
    "wildflower meadow with warm backlight",
    "rainforest trail with soft mist",
    "botanical garden with deep green foliage",
    "quiet riverbank with natural stones",
    "sunlit palm grove",
    "misty hillside garden",
    "wooden nature retreat deck",
    "greenhouse filled with tropical plants",
    "forest waterfall overlook",
    "coastal garden with palms and dune grass",
]

OUTFITS = [
    "tasteful fitted sage green wrap dress",
    "earth-toned linen co-ord outfit",
    "ivory sleeveless jumpsuit with simple gold jewelry",
    "forest green satin blouse and tailored skirt",
    "rust red resort dress with elegant neckline",
    "black fitted editorial outfit with modest coverage",
    "cream crop top with high-waist wide-leg trousers",
    "olive bodycon midi dress with natural fabric texture",
    "terracotta draped dress with clean silhouette",
    "white linen shirt tied at waist with long skirt",
]

BODY_VARIATIONS = [
    "slim athletic figure",
    "soft curvy figure",
    "tall elegant figure",
    "petite curvy figure",
    "toned hourglass figure",
    "natural mature feminine figure",
]

AGES = [
    "adult woman age 30",
    "adult woman age 31",
    "adult woman age 32",
    "adult woman age 33",
    "adult woman age 34",
    "adult woman age 35",
    "adult woman age 36",
    "adult woman age 37",
    "adult woman age 38",
    "adult woman age 39",
]


def build_record(index: int) -> dict[str, str]:
    pose = POSES[index % len(POSES)]
    setting = SETTINGS[(index * 3) % len(SETTINGS)]
    outfit = OUTFITS[(index * 7) % len(OUTFITS)]
    body = BODY_VARIATIONS[(index * 5) % len(BODY_VARIATIONS)]
    age = AGES[(index * 2) % len(AGES)]
    stem = f"vanya_glam_{index + 1:03d}"
    caption = (
        f"{TRIGGER}, original fictional {age}, nature glamour influencer, "
        f"South Asian inspired but not based on any real person, long dark wavy hair, "
        f"expressive brown eyes, sun kissed skin, {body}, {pose}, {setting}, "
        f"{outfit}, hot and sexy fashion editorial energy, tasteful non explicit, "
        f"photorealistic, elegant mature confidence"
    )
    prompt = f"""
Use case: photorealistic-natural
Asset type: LoRA training dataset image
Subject: original fictional adult woman named Vanya, {age}, {body}, South Asian inspired but not based on any real person, long dark wavy hair, expressive brown eyes, sun kissed skin
Pose/posture: {pose}
Scene/backdrop: {setting}
Wardrobe: {outfit}
Style/medium: photorealistic high-end fashion editorial nature photography
Mood: hot, sexy, confident, elegant, mature, aspirational, non-explicit
Composition/framing: varied training-data composition, realistic anatomy, natural hands, clean face detail, full usable body silhouette when applicable
Lighting: warm natural light, cinematic but realistic
Constraints: original fictional person only; do not copy Vrutika Patel, celebrities, influencers, private people, or copyrighted photos
Avoid: nudity, explicit sexual pose, sheer clothing, fetish styling, text, logo, watermark, distorted hands, extra fingers, uncanny face, childlike appearance
""".strip()
    return {
        "index": str(index + 1),
        "stem": stem,
        "image_file": f"{stem}.png",
        "caption_file": f"{stem}.txt",
        "prompt_file": f"{stem}.prompt.txt",
        "caption": caption,
        "prompt": prompt,
    }


def main() -> int:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    records = [build_record(i) for i in range(60)]

    for record in records:
        (DATASET_DIR / record["caption_file"]).write_text(
            record["caption"] + "\n", encoding="utf-8"
        )
        (DATASET_DIR / record["prompt_file"]).write_text(
            record["prompt"] + "\n", encoding="utf-8"
        )

    with (DATASET_DIR / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "stem",
                "image_file",
                "caption_file",
                "prompt_file",
                "caption",
            ],
        )
        writer.writeheader()
        for record in records:
            row = {key: record[key] for key in writer.fieldnames}
            writer.writerow(row)

    with (DATASET_DIR / "prompts.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(
                    {
                        "image_file": record["image_file"],
                        "caption_file": record["caption_file"],
                        "prompt": record["prompt"],
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )

    (DATASET_DIR / "README.md").write_text(
        f"""# Vanya Glam 60 Dataset

This folder contains a 60-image training dataset plan for the original fictional character `Vanya`.

Trigger word:

```text
{TRIGGER}
```

The `.txt` caption files and `.prompt.txt` prompt files are already created. Actual `.png` image files are generated by:

```powershell
cd "C:\\Gen AI\\Human AI"
$env:OPENAI_API_KEY="your_api_key_here"
python training\\scripts\\generate_vanya_glam_dataset.py
```

Rules:

- Original fictional adult character only.
- Age range is 30-39.
- Glamorous, hot, and sexy fashion-editorial energy is allowed here only as tasteful non-explicit styling.
- Do not use or request Vrutika Patel, celebrities, influencers, private people, or copyrighted photos.
- Do not add nudity, explicit poses, sheer clothing, fetish styling, logos, text, or watermarks.

Expected final count after generation:

```text
60 png images
60 txt captions
60 prompt txt files
manifest.csv
prompts.jsonl
```
""",
        encoding="utf-8",
    )
    print(f"Created 60-entry dataset plan at: {DATASET_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
