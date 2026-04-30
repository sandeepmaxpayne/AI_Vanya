from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import imageio.v2 as imageio
import numpy as np
from openai import OpenAI
from PIL import Image, ImageOps
from pydantic import BaseModel, Field
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent


def load_environment() -> None:
    load_dotenv(ROOT / ".env", encoding="utf-8-sig")
    load_dotenv(ROOT / ".env.example", encoding="utf-8-sig")


load_environment()

OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
TRAINED_DATA_DIR = ROOT / "trained_data"
TRAINED_DATA_DIR.mkdir(exist_ok=True)
TRAINED_JSON_PATH = TRAINED_DATA_DIR / "trained_vanya.json"
TRAINED_CACHE_DIR = OUTPUT_DIR / "trained_cache"
TRAINED_CACHE_DIR.mkdir(exist_ok=True)

IMAGE_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1.5")
app = FastAPI(
    title="Original AI Influencer Image API",
    description=(
        "Generates a fictional, rights-safe nature influencer character in many "
        "poses and postures. Do not use this to copy a real person's likeness."
    ),
)
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


def safe_error_text(exc: Exception) -> str:
    text = str(exc) or exc.__class__.__name__
    text = re.sub(r"sk-[A-Za-z0-9_-]+", "sk-***", text)
    return text[:900]


@app.exception_handler(Exception)
async def json_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={
            "detail": f"Internal server error: {safe_error_text(exc)}",
            "path": request.url.path,
        },
    )


CHARACTER_BIBLE = """
Original fictional adult female AI influencer named Ira Vanya.
Nature-forward luxury lifestyle persona, South Asian-inspired but not based on
any real person, expressive brown eyes, long dark wavy hair, warm sun-kissed
skin, natural makeup, confident editorial presence, elegant modern wardrobe.
Tasteful glamorous energy only: no nudity, no explicit pose, no transparent or
fetish clothing, no watermark, no text, no logo.
"""

SAFETY_RULES = """
Create an original fictional person only. Do not copy, imitate, or recreate the
likeness of Vrutika Patel, celebrities, influencers, private people, or any real
person. If a reference image is supplied, use it only as a broad character
consistency reference when the user owns or has licensed it.
"""

BLOCKED_REAL_PERSON_TERMS = {
    "vrutika patel",
}


class GenerateRequest(BaseModel):
    prompt: str = Field(
        default="photorealistic editorial portrait in a lush natural setting",
        max_length=4000,
    )
    pose: str = Field(default="standing relaxed three-quarter pose", max_length=500)
    posture: str = Field(default="confident upright posture", max_length=500)
    setting: str = Field(default="tropical forest edge at golden hour", max_length=500)
    outfit: str = Field(
        default="tasteful fitted nature-inspired fashion outfit",
        max_length=500,
    )
    size: Literal["1024x1024", "1024x1536", "1536x1024", "auto"] = "1024x1536"
    quality: Literal["low", "medium", "high", "auto"] = "medium"
    output_format: Literal["png", "jpeg", "webp"] = "png"


class PosePackRequest(BaseModel):
    poses: list[str] = Field(
        default=[
            "standing relaxed three-quarter portrait",
            "walking through wildflowers, candid editorial",
            "seated on a mossy rock, elegant posture",
            "over-the-shoulder look near tropical leaves",
            "hands gently touching tall grass, soft smile",
        ],
        min_length=1,
        max_length=20,
    )
    setting: str = Field(default="lush nature backdrop at golden hour", max_length=500)
    outfit: str = Field(default="tasteful earth-toned fashion outfit", max_length=500)
    size: Literal["1024x1024", "1024x1536", "1536x1024", "auto"] = "1024x1536"
    quality: Literal["low", "medium", "high", "auto"] = "medium"
    output_format: Literal["png", "jpeg", "webp"] = "png"


class ImageResult(BaseModel):
    prompt: str
    image_path: str


class WebCreateRequest(BaseModel):
    prompt: str = Field(default="Vanya walking through a tropical garden at golden hour")
    output_type: Literal["image", "reel"] = "image"
    pose: str = Field(default="confident fashion editorial pose")
    setting: str = Field(default="lush nature location at golden hour")
    outfit: str = Field(default="tasteful fitted nature-inspired fashion outfit")
    frame_count: int = Field(default=4, ge=2, le=8)
    duration_seconds: int = Field(default=8, ge=3, le=20)
    quality: Literal["low", "medium", "high", "auto"] = "medium"
    use_trained_json: bool = True
    offline_only: bool = False


class WebCreateResult(BaseModel):
    output_type: Literal["image", "reel"]
    prompt: str
    files: list[str]
    url: str
    source: Literal["trained-json", "similar-trained-json", "openai-api"]
    cache_key: str


def reject_real_person_copy(text: str) -> None:
    lowered = text.lower()
    for term in BLOCKED_REAL_PERSON_TERMS:
        if term in lowered:
            raise HTTPException(
                status_code=400,
                detail=(
                    "This API creates an original fictional AI influencer only. "
                    "Remove real-person likeness requests and try again."
                ),
            )


def slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return value[:60] or "image"


def utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def normalize_text(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def request_cache_payload(req: WebCreateRequest) -> dict[str, object]:
    payload: dict[str, object] = {
        "output_type": req.output_type,
        "prompt": normalize_text(req.prompt),
        "pose": normalize_text(req.pose),
        "setting": normalize_text(req.setting),
        "outfit": normalize_text(req.outfit),
    }
    if req.output_type == "reel":
        payload["frame_count"] = req.frame_count
        payload["duration_seconds"] = req.duration_seconds
    return payload


def cache_key_for_request(req: WebCreateRequest) -> str:
    payload = request_cache_payload(req)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def request_search_text(req: WebCreateRequest) -> str:
    return normalize_text(" ".join([req.prompt, req.pose, req.setting, req.outfit]))


def word_set(value: str) -> set[str]:
    stop_words = {
        "a",
        "an",
        "and",
        "at",
        "for",
        "in",
        "of",
        "on",
        "the",
        "to",
        "with",
        "vanya",
        "image",
        "reel",
    }
    return {word for word in normalize_text(value).split() if word not in stop_words}


def similarity_score(left: str, right: str) -> float:
    left_words = word_set(left)
    right_words = word_set(right)
    if not left_words or not right_words:
        return 0.0
    return len(left_words & right_words) / len(left_words | right_words)


def output_url(path: Path) -> str:
    try:
        relative = path.resolve().relative_to(OUTPUT_DIR.resolve())
        return f"/outputs/{relative.as_posix()}"
    except ValueError:
        return f"/outputs/{path.name}"


def output_relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def record_file_exists(record: dict[str, object]) -> bool:
    files = record.get("files")
    if not isinstance(files, list) or not files:
        return False
    for value in files:
        if not isinstance(value, str):
            return False
        if not (ROOT / value).exists():
            return False
    return True


def default_trained_data() -> dict[str, object]:
    return {
        "schema_version": 1,
        "character": "Vanya",
        "trigger_word": "iravanyaai",
        "description": (
            "Local trained JSON cache for original fictional Vanya outputs. "
            "Records are reused before making OpenAI calls."
        ),
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "records": [],
    }


def load_trained_data() -> dict[str, object]:
    if not TRAINED_JSON_PATH.exists():
        data = default_trained_data()
        save_trained_data(data)
        return data
    try:
        data = json.loads(TRAINED_JSON_PATH.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"trained_vanya.json is not valid JSON: {exc}",
        ) from exc
    if not isinstance(data.get("records"), list):
        data["records"] = []
    return data


def save_trained_data(data: dict[str, object]) -> None:
    data["updated_at"] = utc_now()
    TRAINED_JSON_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def seed_trained_json() -> None:
    data = load_trained_data()
    records = data.get("records", [])
    if not isinstance(records, list):
        records = []
        data["records"] = records
    if any(record.get("source") == "seed-reference" for record in records if isinstance(record, dict)):
        return

    source = ROOT / "assets" / "ai_influencer_reference.png"
    if not source.exists():
        return
    target = TRAINED_CACHE_DIR / "vanya_seed_reference.png"
    if not target.exists():
        shutil.copy2(source, target)

    seed_req = WebCreateRequest(
        prompt="Vanya walking through a lush tropical garden at golden hour, glamorous mature nature influencer, confident smile",
        output_type="image",
        pose="confident three-quarter fashion editorial pose",
        setting="lush nature location at golden hour",
        outfit="tasteful fitted nature-inspired fashion outfit",
    )
    final_prompt = build_prompt(
        prompt=seed_req.prompt,
        pose=seed_req.pose,
        posture="confident mature fashion-editorial posture",
        setting=seed_req.setting,
        outfit=seed_req.outfit,
    )
    records.append(
        {
            "id": "seed-reference",
            "cache_key": cache_key_for_request(seed_req),
            "output_type": "image",
            "request": seed_req.model_dump(),
            "search_text": request_search_text(seed_req),
            "final_prompt": final_prompt,
            "files": [output_relative_path(target)],
            "url": output_url(target),
            "source": "seed-reference",
            "created_at": utc_now(),
        }
    )
    save_trained_data(data)


def trained_result_from_record(
    record: dict[str, object],
    *,
    source: Literal["trained-json", "similar-trained-json"],
) -> WebCreateResult:
    files = [str(value) for value in record.get("files", []) if isinstance(value, str)]
    prompt = str(record.get("final_prompt") or record.get("prompt") or "")
    output_type = record.get("output_type")
    if output_type not in {"image", "reel"}:
        output_type = "image"
    url = str(record.get("url") or "")
    if not url and files:
        url = output_url(ROOT / files[-1])
    return WebCreateResult(
        output_type=output_type,  # type: ignore[arg-type]
        prompt=prompt,
        files=files,
        url=url,
        source=source,
        cache_key=str(record.get("cache_key") or ""),
    )


def find_trained_record(req: WebCreateRequest) -> tuple[dict[str, object] | None, str]:
    data = load_trained_data()
    records = data.get("records", [])
    if not isinstance(records, list):
        return None, "miss"

    key = cache_key_for_request(req)
    for record in records:
        if (
            isinstance(record, dict)
            and record.get("cache_key") == key
            and record.get("output_type") == req.output_type
            and record_file_exists(record)
        ):
            return record, "exact"

    search_text = request_search_text(req)
    best_record: dict[str, object] | None = None
    best_score = 0.0
    for record in records:
        if not isinstance(record, dict):
            continue
        if record.get("output_type") != req.output_type or not record_file_exists(record):
            continue
        score = similarity_score(search_text, str(record.get("search_text") or ""))
        if score > best_score:
            best_score = score
            best_record = record

    if best_record is not None and best_score >= 0.86:
        return best_record, "similar"
    return None, "miss"


def add_trained_record(
    req: WebCreateRequest,
    *,
    final_prompt: str,
    files: list[Path],
    url: str,
) -> str:
    data = load_trained_data()
    records = data.get("records", [])
    if not isinstance(records, list):
        records = []
        data["records"] = records
    key = cache_key_for_request(req)
    record = {
        "id": key,
        "cache_key": key,
        "output_type": req.output_type,
        "request": req.model_dump(),
        "search_text": request_search_text(req),
        "final_prompt": final_prompt,
        "files": [output_relative_path(path) for path in files],
        "url": url,
        "source": "openai-api",
        "created_at": utc_now(),
    }

    replaced = False
    for index, existing in enumerate(records):
        if isinstance(existing, dict) and existing.get("cache_key") == key:
            records[index] = record
            replaced = True
            break
    if not replaced:
        records.append(record)
    save_trained_data(data)
    return key


def build_prompt(
    *,
    prompt: str,
    pose: str,
    posture: str,
    setting: str,
    outfit: str,
) -> str:
    reject_real_person_copy(" ".join([prompt, pose, posture, setting, outfit]))
    return f"""
Use case: photorealistic-natural
Asset type: AI-generated influencer image
Character bible: {CHARACTER_BIBLE}
Primary request: {prompt}
Pose: {pose}
Posture: {posture}
Scene/backdrop: {setting}
Wardrobe: {outfit}
Style/medium: photorealistic high-end fashion and nature lifestyle photography
Composition/framing: vertical social-media campaign frame, polished influencer
portrait, realistic anatomy, natural hands, clean depth of field
Lighting/mood: warm golden-hour light, fresh, confident, aspirational
Constraints: {SAFETY_RULES}
Avoid: nudity, explicit sexual content, copied real-person likeness, text, logo,
watermark, distorted hands, extra fingers, plastic skin, uncanny face
""".strip()


seed_trained_json()


def save_b64_image(b64_json: str, name: str, output_format: str) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    path = OUTPUT_DIR / f"{timestamp}-{slugify(name)}.{output_format}"
    path.write_bytes(base64.b64decode(b64_json))
    return path


def create_image(prompt: str, *, size: str, quality: str, output_format: str) -> Path:
    client = OpenAI()
    response = client.images.generate(
        model=IMAGE_MODEL,
        prompt=prompt,
        size=size,
        quality=quality,
        output_format=output_format,
    )
    return save_b64_image(response.data[0].b64_json, prompt, output_format)


def build_reel_frame_prompt(base_prompt: str, frame_number: int, total_frames: int) -> str:
    frame_notes = [
        "opening frame, cinematic vertical portrait, Vanya looking toward camera",
        "gentle movement frame, Vanya walking naturally through the scene",
        "detail frame, confident pose with hair and outfit catching warm light",
        "closing frame, polished influencer reel ending pose with soft smile",
    ]
    note = frame_notes[(frame_number - 1) % len(frame_notes)]
    return f"{base_prompt}\nReel frame {frame_number} of {total_frames}: {note}."


def make_reel_from_images(
    image_paths: list[Path],
    *,
    name: str,
    duration_seconds: int,
    fps: int = 24,
) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    reel_path = OUTPUT_DIR / f"{timestamp}-{slugify(name)}-reel.mp4"
    width, height = 1080, 1920
    total_frames = duration_seconds * fps
    frames_per_image = max(1, total_frames // len(image_paths))

    with imageio.get_writer(
        reel_path,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=16,
    ) as writer:
        written = 0
        for image_index, image_path in enumerate(image_paths):
            image = Image.open(image_path).convert("RGB")
            for local_frame in range(frames_per_image):
                progress = local_frame / max(1, frames_per_image - 1)
                zoom = 1.0 + (0.06 * progress)
                frame_width = int(width * zoom)
                frame_height = int(height * zoom)
                fitted = ImageOps.fit(
                    image,
                    (frame_width, frame_height),
                    method=Image.Resampling.LANCZOS,
                    centering=(0.5, 0.48),
                )
                left = max(0, (frame_width - width) // 2)
                top_shift = int((progress - 0.5) * 28)
                top = max(0, min(frame_height - height, (frame_height - height) // 2 + top_shift))
                frame = fitted.crop((left, top, left + width, top + height))
                writer.append_data(np.asarray(frame))
                written += 1

        while written < total_frames:
            final_frame = ImageOps.fit(
                Image.open(image_paths[-1]).convert("RGB"),
                (width, height),
                method=Image.Resampling.LANCZOS,
                centering=(0.5, 0.48),
            )
            writer.append_data(np.asarray(final_frame))
            written += 1

    return reel_path


def create_reel(req: WebCreateRequest) -> WebCreateResult:
    base_prompt = build_prompt(
        prompt=req.prompt,
        pose=req.pose,
        posture="confident mature fashion-editorial posture",
        setting=req.setting,
        outfit=req.outfit,
    )
    frame_paths: list[Path] = []
    for frame_number in range(1, req.frame_count + 1):
        frame_prompt = build_reel_frame_prompt(base_prompt, frame_number, req.frame_count)
        frame_paths.append(
            create_image(
                frame_prompt,
                size="1024x1536",
                quality=req.quality,
                output_format="png",
            )
        )

    reel_path = make_reel_from_images(
        frame_paths,
        name=req.prompt,
        duration_seconds=req.duration_seconds,
    )
    key = add_trained_record(
        req,
        final_prompt=base_prompt,
        files=[*frame_paths, reel_path],
        url=output_url(reel_path),
    )
    return WebCreateResult(
        output_type="reel",
        prompt=base_prompt,
        files=[str(path) for path in [*frame_paths, reel_path]],
        url=output_url(reel_path),
        source="openai-api",
        cache_key=key,
    )


def recent_outputs() -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for path in sorted(OUTPUT_DIR.glob("*"), key=lambda item: item.stat().st_mtime, reverse=True):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".mp4"}:
            continue
        items.append(
            {
                "name": path.name,
                "url": output_url(path),
                "type": "video" if path.suffix.lower() == ".mp4" else "image",
            }
        )
    return items[:24]


APP_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Vanya Studio</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f4f1eb;
      color: #17201b;
    }
    * { box-sizing: border-box; }
    body { margin: 0; min-height: 100vh; }
    .shell { display: grid; grid-template-columns: 420px 1fr; min-height: 100vh; }
    aside {
      background: #ffffff;
      border-right: 1px solid #ded7ca;
      padding: 28px;
      position: sticky;
      top: 0;
      height: 100vh;
      overflow-y: auto;
    }
    main { padding: 28px; }
    h1 { margin: 0 0 8px; font-size: 28px; line-height: 1.1; letter-spacing: 0; }
    .sub { margin: 0 0 24px; color: #667168; line-height: 1.45; }
    label { display: block; margin: 18px 0 8px; font-weight: 650; font-size: 14px; }
    textarea, input, select {
      width: 100%;
      border: 1px solid #cfc7b8;
      border-radius: 8px;
      padding: 11px 12px;
      font: inherit;
      background: #fffdf9;
      color: #17201b;
    }
    textarea { min-height: 128px; resize: vertical; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .checkline {
      display: flex;
      align-items: center;
      gap: 10px;
      margin: 16px 0 0;
      font-weight: 650;
      color: #26352d;
    }
    .checkline input {
      width: 18px;
      height: 18px;
      margin: 0;
      accent-color: #184d39;
    }
    button {
      width: 100%;
      margin-top: 22px;
      border: 0;
      border-radius: 8px;
      background: #184d39;
      color: white;
      min-height: 46px;
      font: inherit;
      font-weight: 750;
      cursor: pointer;
    }
    button:disabled { opacity: .6; cursor: progress; }
    .status {
      margin-top: 14px;
      min-height: 24px;
      color: #667168;
      line-height: 1.4;
      white-space: pre-wrap;
    }
    .preview {
      min-height: 65vh;
      display: grid;
      place-items: center;
      background: #ebe5db;
      border: 1px solid #ded7ca;
      border-radius: 8px;
      overflow: hidden;
    }
    .preview img, .preview video {
      display: block;
      max-width: 100%;
      max-height: 78vh;
      object-fit: contain;
      background: #111;
    }
    .placeholder { color: #6c746e; text-align: center; max-width: 380px; line-height: 1.5; }
    .gallery {
      margin-top: 24px;
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
      gap: 12px;
    }
    .tile {
      border: 1px solid #ded7ca;
      border-radius: 8px;
      overflow: hidden;
      background: #fff;
      min-height: 220px;
    }
    .tile img, .tile video {
      width: 100%;
      aspect-ratio: 9 / 14;
      object-fit: cover;
      display: block;
      background: #111;
    }
    .tile a {
      display: block;
      padding: 10px;
      color: #184d39;
      text-decoration: none;
      font-size: 13px;
      overflow-wrap: anywhere;
    }
    @media (max-width: 860px) {
      .shell { grid-template-columns: 1fr; }
      aside { position: static; height: auto; }
      main { padding: 18px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <aside>
      <h1>Vanya Studio</h1>
      <p class="sub">Create original Vanya images or short vertical reels from your prompt.</p>
      <form id="createForm">
        <label for="prompt">Prompt</label>
        <textarea id="prompt" name="prompt">Vanya walking through a lush tropical garden at golden hour, glamorous mature nature influencer, confident smile</textarea>

        <div class="row">
          <div>
            <label for="output_type">Output</label>
            <select id="output_type" name="output_type">
              <option value="image">Image</option>
              <option value="reel">Short reel</option>
            </select>
          </div>
          <div>
            <label for="quality">Quality</label>
            <select id="quality" name="quality">
              <option value="medium">Medium</option>
              <option value="low">Low</option>
              <option value="high">High</option>
              <option value="auto">Auto</option>
            </select>
          </div>
        </div>

        <label for="pose">Pose</label>
        <input id="pose" name="pose" value="confident three-quarter fashion editorial pose" />

        <label for="setting">Setting</label>
        <input id="setting" name="setting" value="lush nature location at golden hour" />

        <label for="outfit">Outfit</label>
        <input id="outfit" name="outfit" value="tasteful fitted nature-inspired fashion outfit" />

        <div class="row">
          <div>
            <label for="frame_count">Reel frames</label>
            <input id="frame_count" name="frame_count" type="number" min="2" max="8" value="4" />
          </div>
          <div>
            <label for="duration_seconds">Seconds</label>
            <input id="duration_seconds" name="duration_seconds" type="number" min="3" max="20" value="8" />
          </div>
        </div>

        <label class="checkline">
          <input id="use_trained_json" name="use_trained_json" type="checkbox" checked />
          <span>Use trained JSON</span>
        </label>

        <label class="checkline">
          <input id="offline_only" name="offline_only" type="checkbox" />
          <span>Offline only</span>
        </label>

        <button id="submitButton" type="submit">Create</button>
        <div id="status" class="status"></div>
      </form>
    </aside>
    <main>
      <section id="preview" class="preview">
        <div class="placeholder">Your image or short reel will appear here after generation.</div>
      </section>
      <section id="gallery" class="gallery"></section>
    </main>
  </div>
  <script>
    const form = document.getElementById("createForm");
    const statusBox = document.getElementById("status");
    const button = document.getElementById("submitButton");
    const preview = document.getElementById("preview");
    const gallery = document.getElementById("gallery");

    function renderAsset(container, item, full = false) {
      const media = item.output_type === "reel" || item.type === "video"
        ? document.createElement("video")
        : document.createElement("img");
      media.src = item.url;
      if (media.tagName === "VIDEO") {
        media.controls = true;
        media.loop = true;
        media.playsInline = true;
      }
      container.innerHTML = "";
      container.appendChild(media);
      if (full && media.tagName === "VIDEO") media.play().catch(() => {});
    }

    async function readJsonResponse(res) {
      const text = await res.text();
      let payload = {};
      try {
        payload = text ? JSON.parse(text) : {};
      } catch {
        throw new Error(text || `HTTP ${res.status}`);
      }
      if (!res.ok) {
        throw new Error(payload.detail || payload.message || `HTTP ${res.status}`);
      }
      return payload;
    }

    async function loadGallery() {
      const res = await fetch("/gallery");
      const items = await readJsonResponse(res);
      gallery.innerHTML = "";
      for (const item of items) {
        const tile = document.createElement("article");
        tile.className = "tile";
        renderAsset(tile, item);
        const link = document.createElement("a");
        link.href = item.url;
        link.textContent = item.name;
        link.target = "_blank";
        tile.appendChild(link);
        gallery.appendChild(tile);
      }
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      button.disabled = true;
      statusBox.textContent = "Creating Vanya output. Reels take longer because they generate multiple frames.";
      const data = Object.fromEntries(new FormData(form).entries());
      data.frame_count = Number(data.frame_count);
      data.duration_seconds = Number(data.duration_seconds);
      data.use_trained_json = document.getElementById("use_trained_json").checked;
      data.offline_only = document.getElementById("offline_only").checked;
      try {
        const res = await fetch("/web/create", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(data)
        });
        const result = await readJsonResponse(res);
        renderAsset(preview, result, true);
        const sourceLabel = result.source === "openai-api"
          ? "Generated via API and saved to trained JSON"
          : "Loaded from trained JSON";
        statusBox.textContent = `${sourceLabel}\n${result.output_type}: ${result.url}`;
        await loadGallery();
      } catch (error) {
        statusBox.textContent = error.message;
      } finally {
        button.disabled = false;
      }
    });

    loadGallery();
  </script>
</body>
</html>
"""


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "image_model": IMAGE_MODEL,
        "api_key_loaded": str(bool(os.getenv("OPENAI_API_KEY"))).lower(),
    }


@app.get("/", response_class=HTMLResponse)
def web_app() -> str:
    return APP_HTML


@app.get("/gallery")
def gallery() -> list[dict[str, str]]:
    return recent_outputs()


@app.get("/trained-json")
def trained_json() -> dict[str, object]:
    return load_trained_data()


@app.get("/trained-json/status")
def trained_json_status() -> dict[str, object]:
    data = load_trained_data()
    records = data.get("records", [])
    if not isinstance(records, list):
        records = []
    usable = [record for record in records if isinstance(record, dict) and record_file_exists(record)]
    return {
        "trained_json": str(TRAINED_JSON_PATH),
        "records": len(records),
        "usable_records": len(usable),
        "api_key_loaded": bool(os.getenv("OPENAI_API_KEY")),
    }


@app.post("/web/create", response_model=WebCreateResult)
def web_create(req: WebCreateRequest) -> WebCreateResult:
    if req.use_trained_json:
        record, match_type = find_trained_record(req)
        if record is not None:
            return trained_result_from_record(
                record,
                source="trained-json" if match_type == "exact" else "similar-trained-json",
            )

    if req.offline_only:
        raise HTTPException(
            status_code=404,
            detail=(
                "No trained JSON match found for this input. Disable Offline only "
                "to generate it once and update the trained JSON."
            ),
        )

    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=400,
            detail=(
                "OPENAI_API_KEY is not set and no trained JSON match was found. "
                "Add this prompt online once, or use an existing trained JSON prompt."
            ),
        )

    if req.output_type == "reel":
        return create_reel(req)

    final_prompt = build_prompt(
        prompt=req.prompt,
        pose=req.pose,
        posture="confident mature fashion-editorial posture",
        setting=req.setting,
        outfit=req.outfit,
    )
    image_path = create_image(
        final_prompt,
        size="1024x1536",
        quality=req.quality,
        output_format="png",
    )
    key = add_trained_record(
        req,
        final_prompt=final_prompt,
        files=[image_path],
        url=output_url(image_path),
    )
    return WebCreateResult(
        output_type="image",
        prompt=final_prompt,
        files=[str(image_path)],
        url=output_url(image_path),
        source="openai-api",
        cache_key=key,
    )


@app.post("/generate", response_model=ImageResult)
def generate(req: GenerateRequest) -> ImageResult:
    final_prompt = build_prompt(
        prompt=req.prompt,
        pose=req.pose,
        posture=req.posture,
        setting=req.setting,
        outfit=req.outfit,
    )
    image_path = create_image(
        final_prompt,
        size=req.size,
        quality=req.quality,
        output_format=req.output_format,
    )
    return ImageResult(prompt=final_prompt, image_path=str(image_path))


@app.post("/pose-pack", response_model=list[ImageResult])
def pose_pack(req: PosePackRequest) -> list[ImageResult]:
    results: list[ImageResult] = []
    for pose in req.poses:
        final_prompt = build_prompt(
            prompt="consistent original AI influencer character image",
            pose=pose,
            posture=pose,
            setting=req.setting,
            outfit=req.outfit,
        )
        image_path = create_image(
            final_prompt,
            size=req.size,
            quality=req.quality,
            output_format=req.output_format,
        )
        results.append(ImageResult(prompt=final_prompt, image_path=str(image_path)))
    return results


@app.post("/generate-with-reference", response_model=ImageResult)
async def generate_with_reference(
    image: UploadFile = File(...),
    prompt: str = Form("new pose variation of the same original fictional character"),
    pose: str = Form("standing relaxed three-quarter pose"),
    posture: str = Form("confident upright posture"),
    setting: str = Form("lush nature backdrop at golden hour"),
    outfit: str = Form("tasteful nature-inspired fashion outfit"),
    size: Literal["1024x1024", "1024x1536", "1536x1024", "auto"] = Form("1024x1536"),
    quality: Literal["low", "medium", "high", "auto"] = Form("medium"),
    output_format: Literal["png", "jpeg", "webp"] = Form("png"),
) -> ImageResult:
    final_prompt = build_prompt(
        prompt=prompt,
        pose=pose,
        posture=posture,
        setting=setting,
        outfit=outfit,
    )
    suffix = Path(image.filename or "reference.png").suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp.write(await image.read())
        temp_path = Path(temp.name)

    try:
        client = OpenAI()
        with temp_path.open("rb") as reference:
            response = client.images.edit(
                model=IMAGE_MODEL,
                image=reference,
                prompt=final_prompt,
                size=size,
                quality=quality,
                output_format=output_format,
            )
        image_path = save_b64_image(response.data[0].b64_json, prompt, output_format)
        return ImageResult(prompt=final_prompt, image_path=str(image_path))
    finally:
        temp_path.unlink(missing_ok=True)
