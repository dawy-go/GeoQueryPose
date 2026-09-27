param(
    [Parameter(Mandatory = $true)]
    [string]$DataRoot,
    [string]$Gpus = "0",
    [string]$Python = "",
    [string]$Note = "final-real275-rotation-safe",
    [int]$MaxTestImages = 0
)

$ErrorActionPreference = "Stop"
$packageRoot = $PSScriptRoot
$resolvedDataRoot = (Resolve-Path -LiteralPath $DataRoot).Path
$detectionDir = Join-Path $resolvedDataRoot "segmentation_results\REAL275_groundingdino_sam_full_bottle_to_can"

if (-not (Test-Path -LiteralPath (Join-Path $resolvedDataRoot "Real\test"))) {
    throw "REAL275 test images were not found under: $resolvedDataRoot"
}
if (-not (Test-Path -LiteralPath $detectionDir)) {
    throw "Final detection input was not found under: $detectionDir"
}

if (-not $Python) {
    $repoPython = Join-Path $packageRoot ".venv\Scripts\python.exe"
    $Python = if (Test-Path -LiteralPath $repoPython) { $repoPython } else { "python" }
}

$testArguments = @(
    "test.py",
    "--gpus", $Gpus,
    "--config", "configs/final-real275-rotation-safe.yaml",
    "--data-dir", $resolvedDataRoot,
    "--segmentation-results-dir", $detectionDir,
    "--note", $Note
)
if ($MaxTestImages -gt 0) {
    $testArguments += @("--max-test-images", $MaxTestImages)
}

Push-Location $packageRoot
try {
    & $Python @testArguments
    if ($LASTEXITCODE -ne 0) {
        throw "test.py exited with code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
