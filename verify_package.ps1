param(
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"
$packageRoot = $PSScriptRoot
$expectedFiles = @(
    "configs\final-real275-rotation-safe.yaml",
    "test.py",
    "model\da2_joint_geometry_pose.py",
    "model\da2_metric_center_z_pose.py",
    "model\da2_e2e_pose.py",
    "tools\generate_dinov2_nyu_depth.py",
    "tools\generate_nocs_groundingdino_sam_results.py",
    "tools\apply_nocs_reclassifier.py",
    "checkpoints\epoch_30.pth",
    "checkpoints\nocs-reclassifier-stage1-best.pth"
)
$expectedHashes = @{
    "checkpoints\epoch_30.pth" = "F12BA75A7B8FAD71F06CFC714566779F857C4828354E35C1ADD5557C76B04D0D"
    "checkpoints\nocs-reclassifier-stage1-best.pth" = "0B4D247FBC362FD483B93B407F8C4D22E671585076E02E3EEE94C8DF212C90B7"
}

foreach ($relativePath in $expectedFiles) {
    $path = Join-Path $packageRoot $relativePath
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Missing package file: $relativePath"
    }
}
foreach ($entry in $expectedHashes.GetEnumerator()) {
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $packageRoot $entry.Key)).Hash
    if ($actual -ne $entry.Value) {
        throw "SHA-256 mismatch for $($entry.Key): $actual"
    }
}

if (-not $Python) {
    $repoPython = Join-Path $packageRoot ".venv\Scripts\python.exe"
    $Python = if (Test-Path -LiteralPath $repoPython) { $repoPython } else { "python" }
}

Push-Location $packageRoot
try {
    & $Python -c "from utils.config_utils import load_config; c=load_config('configs/final-real275-rotation-safe.yaml'); assert c.checkpoint == 'checkpoints/epoch_30.pth'; assert 'extends:' not in open('configs/final-real275-rotation-safe.yaml', encoding='utf-8').read()"
    if ($LASTEXITCODE -ne 0) {
        throw "Standalone config validation failed"
    }
}
finally {
    Pop-Location
}

Write-Output "Package verification passed: $packageRoot"
