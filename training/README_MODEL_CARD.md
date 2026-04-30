# Ira Vanya SDXL LoRA Model Card

## Model

`ira_vanya_sdxl_lora`

## Character

Ira Vanya is an original fictional adult AI influencer character designed for nature, wellness, and fashion-editorial imagery.

## Intended Use

- Fictional AI influencer portraits
- Nature lifestyle campaign imagery
- Consistent character generation from owned or licensed training material

## Not Allowed

- Do not use this model to impersonate or imitate Vrutika Patel or any real person.
- Do not train on copyrighted photos unless you have permission.
- Do not generate explicit sexual content.
- Do not present the character as a real human.

## Training Data

Training data should contain only owned, licensed, or AI-generated images. Each image should have a matching `.txt` caption containing the trigger word:

```text
iravanyaai
```

## Recommended Dataset

20-60 images:

- 5-10 close-up portraits
- 5-10 waist-up portraits
- 5-10 full-body images
- 5-10 seated, walking, side-profile, and over-shoulder poses
- 5-10 varied nature backgrounds and lighting conditions

## Output File

```text
training/models/ira_vanya_sdxl_lora.safetensors
```
