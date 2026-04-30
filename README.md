# Original AI Influencer Image API

This project creates a fictional, rights-safe nature influencer character. It now runs local-first with a trained JSON/offline model so repeated or new prompts can work without OpenAI billing.

To run: 
```python -m uvicorn influencer_api:app --host 127.0.0.1 --port 8001  ```

## Setup

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
uvicorn influencer_api:app --reload --host 127.0.0.1 --port 8000
```

Open the API docs at:

```text
http://127.0.0.1:8000/docs
```

Open the Vanya web app at:

```text
http://127.0.0.1:8000/
```

The web app lets you enter a prompt and choose either `Image` or `Short reel`. By default it uses the offline local model, creates media from local assets, and saves the prompt/output into `trained_data/trained_vanya.json`.

OpenAI calls are disabled by default:

```text
ALLOW_OPENAI_API=false
```

Only set `ALLOW_OPENAI_API=true` if you intentionally want to spend OpenAI API credits.

## Trained JSON Cache / Offline Reuse

The app now keeps a local trained JSON cache here:

```text
trained_data/trained_vanya.json
```

When `Use trained JSON` is enabled in the web app, the app checks this file first. If the same prompt/settings already exist, it returns the saved image or reel offline with no token use.

If the input is new, the offline local model creates an image or reel from local assets and saves the result into `trained_vanya.json`. The next same prompt is served from JSON.

The offline model profile is here:

```text
trained_data/vanya_offline_model.json
```

Useful endpoints:

```text
http://127.0.0.1:8000/trained-json/status
http://127.0.0.1:8000/trained-json
http://127.0.0.1:8000/trained-json/training-map
http://127.0.0.1:8000/offline-model
```

Every new prompt now updates the trained JSON with:

- generated local image or reel files
- detected position, such as `standing`, `walking`, `seated`, `over shoulder`
- detected posture, such as `confident`, `relaxed`, `hands on waist`
- detected structure, such as `portrait`, `full-body`, `waist-up`, `curvy`
- setting, outfit, mood, safety metadata, and a LoRA-style training caption

## Generate One Image

```powershell
Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/generate" `
  -ContentType "application/json" `
  -Body '{
    "prompt": "photorealistic fashion editorial image for Instagram",
    "pose": "walking through a forest path, candid smile",
    "posture": "relaxed confident posture",
    "setting": "misty green tea garden at sunrise",
    "outfit": "tasteful fitted ivory linen outfit with earthy accessories",
    "size": "1024x1536",
    "quality": "medium",
    "output_format": "png"
  }'
```

## Generate Many Poses

```powershell
Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/pose-pack" `
  -ContentType "application/json" `
  -Body '{
    "poses": [
      "standing three-quarter portrait beside tropical leaves",
      "seated on a rock near a waterfall, elegant posture",
      "walking through wildflowers, candid editorial",
      "over-the-shoulder look in a forest trail",
      "hands gently touching tall grass, soft smile"
    ],
    "setting": "lush green nature environment at golden hour",
    "outfit": "tasteful earth-toned fashion outfit",
    "size": "1024x1536",
    "quality": "medium",
    "output_format": "png"
  }'
```

## Generate With The Local Reference Image

The current generated concept image is saved at:

```text
assets/ai_influencer_reference.png
```

Use this only as an original, generated character reference:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/generate-with-reference" `
  -F "image=@assets/ai_influencer_reference.png" `
  -F "prompt=new consistent pose variation of the original fictional nature influencer" `
  -F "pose=standing beside tropical leaves, hand on waist, editorial expression" `
  -F "posture=confident upright posture" `
  -F "setting=warm forest garden at golden hour" `
  -F "outfit=tasteful fitted sage green fashion outfit" `
  -F "size=1024x1536" `
  -F "quality=medium" `
  -F "output_format=png"
```

Generated images are written to `outputs/`.

## Train A Reusable Character LoRA

I added a trainable SDXL LoRA package under:

```text
training/
```

Start here:

```text
training/TRAINING.md
```

The trigger word is:

```text
iravanyaai
```

This is for an original fictional character only. A useful trained model needs 20-60 owned, licensed, or AI-generated images; the included image is only a seed reference.

The 60-image Vanya dataset has its own training config:

```text
training/configs/vanya_glam_60_sdxl_lora.toml
```

After generating the PNGs and installing Kohya SS, train it with:

```powershell
powershell -ExecutionPolicy Bypass -File training\scripts\train_vanya_glam_60.ps1 -KohyaPath "C:\path\to\kohya_ss"
```

## Notes

- Use owned, licensed, or AI-generated reference images only.
- Do not request a real person's likeness.
- This is prompt-based generation and reference-guided editing, not model training.
