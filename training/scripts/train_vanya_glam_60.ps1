param(
    [string]$KohyaPath = "",
    [switch]$GenerateMissingImages
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$DatasetDir = Join-Path $ProjectRoot "training\dataset\vanya_glam_60"
$ConfigPath = Join-Path $ProjectRoot "training\configs\vanya_glam_60_sdxl_lora.toml"
$Generator = Join-Path $ProjectRoot "training\scripts\generate_vanya_glam_dataset.py"

if (!(Test-Path $DatasetDir)) {
    throw "Dataset folder is missing: $DatasetDir"
}

$imageCount = @(Get-ChildItem -Path $DatasetDir -Filter "*.png" -File).Count

if ($imageCount -lt 60) {
    if ($GenerateMissingImages) {
        if (!$env:OPENAI_API_KEY) {
            throw "OPENAI_API_KEY is not set. Set it before generating the missing dataset images."
        }
        python $Generator
        $imageCount = @(Get-ChildItem -Path $DatasetDir -Filter "*.png" -File).Count
    }
}

if ($imageCount -lt 20) {
    throw "Only $imageCount PNG images found. Add or generate at least 20 images before training. 60 is recommended."
}

python (Join-Path $ProjectRoot "training\scripts\validate_dataset.py") $DatasetDir

$accelerate = Get-Command accelerate -ErrorAction SilentlyContinue
if (!$accelerate) {
    throw "accelerate is not installed or not on PATH. Install Kohya SS dependencies before training."
}

if ($KohyaPath) {
    $KohyaRoot = Resolve-Path $KohyaPath
    $TrainScript = Join-Path $KohyaRoot "sdxl_train_network.py"
    if (!(Test-Path $TrainScript)) {
        throw "sdxl_train_network.py not found in Kohya path: $KohyaRoot"
    }
    Push-Location $KohyaRoot
    try {
        accelerate launch .\sdxl_train_network.py --config_file $ConfigPath
    }
    finally {
        Pop-Location
    }
}
else {
    $trainScript = Get-Command sdxl_train_network.py -ErrorAction SilentlyContinue
    if (!$trainScript) {
        throw "sdxl_train_network.py is not on PATH. Re-run with -KohyaPath 'C:\path\to\kohya_ss'."
    }
    accelerate launch sdxl_train_network.py --config_file $ConfigPath
}
