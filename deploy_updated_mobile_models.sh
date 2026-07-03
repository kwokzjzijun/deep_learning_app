#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/Users/gzj/Desktop/YOLO_Project"
APP_ROOT="/Users/gzj/deep_learning_app"
ASSETS_DIR="$APP_ROOT/app/src/main/assets"
YOLO_EXPORT_ENV="/tmp/yolo_mobile_export_env_pure"

URIC_ACID_CHECKPOINT="/Users/gzj/Desktop/YOLO_Project/Results_Opt/ConvNext_2/best_convnext_2_0425.pt"
YOLO_WEIGHTS="/Users/gzj/Desktop/YOLO_Project/YOLO_Seg/best_YOLO11n-seg_0420.pt"
RUN_URIC_ACID=1
RUN_YOLO=1

URIC_ACID_MOBILE=""
YOLO_MOBILE="$PROJECT_ROOT/best_YOLO11n_seg_0420_mobile.ptl"
URIC_ACID_ALIAS="$ASSETS_DIR/uric_acid_current_mobile.ptl"
YOLO_ALIAS="$ASSETS_DIR/yolo_locate_current_mobile.ptl"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --uric-acid-checkpoint|--resnet-checkpoint|--convnext-checkpoint)
      URIC_ACID_CHECKPOINT="$2"
      shift 2
      ;;
    --yolo-weights)
      YOLO_WEIGHTS="$2"
      shift 2
      ;;
    --resnet-only)
      RUN_URIC_ACID=1
      RUN_YOLO=0
      shift
      ;;
    --uric-acid-only|--convnext-only)
      RUN_URIC_ACID=1
      RUN_YOLO=0
      shift
      ;;
    --yolo-only)
      RUN_URIC_ACID=0
      RUN_YOLO=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [--uric-acid-checkpoint PATH] [--yolo-weights PATH] [--uric-acid-only|--yolo-only]" >&2
      exit 1
      ;;
  esac
done

mkdir -p "$ASSETS_DIR"

if [[ "$RUN_URIC_ACID" -eq 1 ]]; then
  if [[ -z "$URIC_ACID_MOBILE" ]]; then
    checkpoint_name="$(basename "$URIC_ACID_CHECKPOINT")"
    checkpoint_stem="${checkpoint_name%.*}"
    URIC_ACID_MOBILE="$PROJECT_ROOT/${checkpoint_stem}_mobile.ptl"
  fi
  python "$PROJECT_ROOT/export_resnetcbam_to_mobile.py" --checkpoint "$URIC_ACID_CHECKPOINT" --out "$URIC_ACID_MOBILE" --copy-assets "$ASSETS_DIR"
  cp -f "$ASSETS_DIR/$(basename "$URIC_ACID_MOBILE")" "$URIC_ACID_ALIAS"
fi

BASE_SITE_PACKAGES="$(python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"

if [[ "$RUN_YOLO" -eq 1 ]]; then
  if [ ! -x "$YOLO_EXPORT_ENV/bin/python" ]; then
    python -m venv "$YOLO_EXPORT_ENV"
    "$YOLO_EXPORT_ENV/bin/python" -m pip install --upgrade pip >/dev/null
    "$YOLO_EXPORT_ENV/bin/python" -m pip install --no-deps ultralytics >/dev/null
  fi

  PYTHONPATH="$BASE_SITE_PACKAGES" "$YOLO_EXPORT_ENV/bin/python" \
    "$PROJECT_ROOT/export_yolo_to_mobile.py" \
    --weights "$YOLO_WEIGHTS" \
    --out "$YOLO_MOBILE" \
    --copy-assets "$ASSETS_DIR"

  cp -f "$ASSETS_DIR/$(basename "$YOLO_MOBILE")" "$YOLO_ALIAS"
fi

ls -lh \
  "$ASSETS_DIR"/*.ptl
echo "[OK] Mobile models updated in $ASSETS_DIR"
