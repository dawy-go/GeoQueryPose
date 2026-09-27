#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash tools/run_nocs_reclassifier_stage1.sh /path/to/NOCS/data" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

DATA_ROOT="$1"
GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_DET_DIR="${TRAIN_DET_DIR:-$DATA_ROOT/segmentation_results/REAL_train_groundingdino_sam}"
MANIFEST="${MANIFEST:-reclassifier_data/real_train_bottle_can_mug.jsonl}"
CLS_LOG="${CLS_LOG:-training_logs/nocs-reclassifier-stage1}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SKIP_DETECTION="${SKIP_DETECTION:-0}"
SAM_MODEL="${SAM_MODEL:-facebook/sam-vit-base}"
MAX_DETECTIONS="${MAX_DETECTIONS:-16}"
SAM_BOX_BATCH_SIZE="${SAM_BOX_BATCH_SIZE:-4}"
GENERATOR_AMP="${GENERATOR_AMP:-1}"
DISABLE_CUDNN="${DISABLE_CUDNN:-0}"

if [[ "$GENERATOR_AMP" == "1" ]]; then
  AMP_FLAG="--amp"
else
  AMP_FLAG="--no-amp"
fi

if [[ "$DISABLE_CUDNN" == "1" ]]; then
  CUDNN_FLAG="--disable-cudnn"
else
  CUDNN_FLAG="--no-disable-cudnn"
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"

echo "[stage 1/3] Generate REAL train detections and predicted masks"
if [[ "$SKIP_DETECTION" == "1" ]]; then
  echo "Skipping detection generation; using $TRAIN_DET_DIR"
else
  "$PYTHON_BIN" tools/generate_nocs_groundingdino_sam_results.py \
    --data-root "$DATA_ROOT" \
    --split-file Real/train_list.txt \
    --output-dir "$TRAIN_DET_DIR" \
    --device cuda:0 \
    --sam-model "$SAM_MODEL" \
    --max-detections "$MAX_DETECTIONS" \
    --sam-box-batch-size "$SAM_BOX_BATCH_SIZE" \
    "$AMP_FLAG" \
    "$CUDNN_FLAG"
fi

echo "[stage 2/3] Build class-agnostic IoU manifest"
"$PYTHON_BIN" tools/build_nocs_reclassifier_manifest.py \
  --data-root "$DATA_ROOT" \
  --detections-dir "$TRAIN_DET_DIR" \
  --output "$MANIFEST" \
  --min-iou 0.5 \
  --val-scenes scene_1

echo "[stage 3/3] Train Bottle/Can/Mug reclassifier"
"$PYTHON_BIN" tools/train_nocs_reclassifier.py \
  --manifest "$MANIFEST" \
  --data-root "$DATA_ROOT" \
  --detections-dir "$TRAIN_DET_DIR" \
  --output-dir "$CLS_LOG" \
  --device cuda:0 \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS"

echo "Stage 1 complete"
echo "Best checkpoint: $CLS_LOG/best.pth"
echo "Training history: $CLS_LOG/history.json"
