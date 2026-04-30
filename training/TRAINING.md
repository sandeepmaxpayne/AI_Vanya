# Training The Original Character Model

This folder is a trainable SDXL LoRA package for the fictional AI influencer character `Ira Vanya`.

Trigger word:

```text
iravanyaai
```

Use the trigger word in every caption and later in every generation prompt.

## What You Have Now

The seed reference image is copied here:

```text
training/dataset/ira_vanya/ira_vanya_001.png
```

This is only a seed. A useful character LoRA usually needs 20-60 owned, licensed, or AI-generated images across different poses, outfits, framings, and lighting. One image will overfit and produce weak results.

## 60-Image Generated Dataset Pack

I also added a 60-image dataset plan for a more mature, glamorous fictional Vanya character:

```text
training/dataset/vanya_glam_60/
```

Create or refresh the captions and prompts:

```powershell
cd "C:\Gen AI\Human AI"
python training\scripts\create_vanya_glam_dataset_plan.py
```

Generate the 60 PNG images after setting `OPENAI_API_KEY`:

```powershell
cd "C:\Gen AI\Human AI"
$env:OPENAI_API_KEY="your_api_key_here"
python training\scripts\generate_vanya_glam_dataset.py
```

Train that 60-image dataset with the dedicated config:

```powershell
cd "C:\Gen AI\Human AI"
powershell -ExecutionPolicy Bypass -File training\scripts\train_vanya_glam_60.ps1 -KohyaPath "C:\path\to\kohya_ss"
```

If the PNG files have not been generated yet, this launcher can generate missing images first:

```powershell
cd "C:\Gen AI\Human AI"
$env:OPENAI_API_KEY="your_api_key_here"
powershell -ExecutionPolicy Bypass -File training\scripts\train_vanya_glam_60.ps1 -GenerateMissingImages -KohyaPath "C:\path\to\kohya_ss"
```

Expected output:

```text
training/models/vanya_glam_60_sdxl_lora.safetensors
```

## Dataset Rules

- Use only images you generated, photographed, or licensed.
- Do not train on Vrutika Patel, celebrities, influencers, private people, or copyrighted photos.
- Keep the character fictional and consistent.
- Add one `.txt` caption beside every image.
- Every caption should include `iravanyaai`.

Good caption example:

```text
iravanyaai, original fictional adult woman, nature lifestyle influencer, long dark wavy hair, expressive brown eyes, full body standing pose, tropical forest path, golden hour light, tasteful fitted sage outfit, non explicit
```

## Validate Dataset

```powershell
cd "C:\Gen AI\Human AI"
python training\scripts\validate_dataset.py
```

## Create Missing Caption Stubs

```powershell
cd "C:\Gen AI\Human AI"
python training\scripts\generate_caption_stubs.py
```

Then edit the generated `.txt` files so each caption describes the actual pose, clothing, framing, and setting.

## Train With Kohya SS

Install Kohya SS separately, then run its SDXL training script from the Kohya folder while pointing to this config:

```powershell
accelerate launch sdxl_train_network.py --config_file "C:\Gen AI\Human AI\training\configs\ira_vanya_sdxl_lora.toml"
```

Expected output:

```text
training/models/ira_vanya_sdxl_lora.safetensors
```

## Prompt After Training

```text
iravanyaai, original fictional adult woman, nature lifestyle influencer, standing in a lush forest at golden hour, tasteful earth toned fashion outfit, photorealistic editorial portrait, non explicit
```

Negative prompt:

```text
real person likeness, celebrity, nudity, explicit pose, watermark, logo, text, distorted hands, extra fingers, uncanny face
```
