# GeoQueryPose

## Prepare REAL275

Set the repository, dataset, and DINOv2 paths:

```powershell
$RepoRoot = "D:\IIM\GeoQueryPose-release"
$DataRoot = "Y:\Datasets\NOCS\data"
$DinoV2Root = "D:\path\to\dinov2"

Set-Location $RepoRoot
```

Place `epoch_30.pth` and `nocs-reclassifier-stage1-best.pth` in `checkpoints/`.

The dataset root must contain the REAL275 images, the test split, and the original result PKLs used for ground-truth metadata:

```text
<DataRoot>/
|-- Real/
|   |-- test/
|   |   `-- scene_*/*_color.png
|   `-- test_list_all.txt
`-- segmentation_results/
    `-- REAL275/results_*.pkl
```

Each PKL in `segmentation_results/REAL275` must contain `gt_class_ids`, `gt_bboxes`, `gt_RTs`, `gt_scales`, and `gt_handle_visibility`.

Generate the metric depth maps:

```powershell
python tools\generate_dinov2_nyu_depth.py `
  --dataset-root $DataRoot `
  --source real-test `
  --dinov2-root $DinoV2Root `
  --backbone base `
  --device cuda:0
```

Generate the GroundingDINO and SAM detections:

```powershell
$MetadataDir = Join-Path $DataRoot "segmentation_results\REAL275"
$DetectionDir = Join-Path $DataRoot "segmentation_results\REAL275_groundingdino_sam_full"

python tools\generate_nocs_groundingdino_sam_results.py `
  --data-root $DataRoot `
  --split-file Real/test_list_all.txt `
  --output-dir $DetectionDir `
  --metadata-dir $MetadataDir `
  --require-gt-metadata `
  --device cuda:0 `
  --sam-model facebook/sam-vit-base `
  --max-detections 16 `
  --sam-box-batch-size 4
```

Apply the Bottle-to-Can reclassifier:

```powershell
$FinalDetectionDir = Join-Path $DataRoot "segmentation_results\REAL275_groundingdino_sam_full_bottle_to_can"

python tools\apply_nocs_reclassifier.py `
  --input-dir $DetectionDir `
  --output-dir $FinalDetectionDir `
  --checkpoint checkpoints\nocs-reclassifier-stage1-best.pth `
  --data-root $DataRoot `
  --device cuda:0 `
  --candidate-class-ids 1 `
  --allowed-transitions bottle:can `
  --min-confidence 0.60 `
  --min-margin 0.10 `
  --score-mode preserve
```

## Run the test

Run one image first:

```powershell
& .\run_test.ps1 -DataRoot $DataRoot -Gpus 0 -MaxTestImages 1 -Note quick-test
```

Run the full REAL275 test:

```powershell
& .\run_test.ps1 -DataRoot $DataRoot -Gpus 0
```
