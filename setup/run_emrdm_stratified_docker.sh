#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${EMRDM_IMAGE:-allclear-emrdm:cu121}"

DATASET_FPATH="${DATASET_FPATH:-setup/emrdm_samples_subset.json}"
EMRDM_PAIRS_FPATH="${EMRDM_PAIRS_FPATH:-setup/emrdm_pairs_0_100.json}"
EMRDM_CONFIG_FPATH="${EMRDM_CONFIG_FPATH:-models/EMRDM/configs/example_training/sentinel.yaml}"
EMRDM_CKPT_FPATH="${EMRDM_CKPT_FPATH:-models/EMRDM/checkpoints/sen12mscr.ckpt}"
BATCH_SIZE="${BATCH_SIZE:-1}"

if [[ ! -f "$ROOT_DIR/$DATASET_FPATH" ]]; then
  echo "[error] Missing dataset JSON: $DATASET_FPATH"
  exit 1
fi
if [[ ! -f "$ROOT_DIR/$EMRDM_PAIRS_FPATH" ]]; then
  echo "[error] Missing EMRDM pairs JSON: $EMRDM_PAIRS_FPATH"
  exit 1
fi
if [[ ! -f "$ROOT_DIR/$EMRDM_CONFIG_FPATH" ]]; then
  echo "[error] Missing EMRDM config: $EMRDM_CONFIG_FPATH"
  exit 1
fi
if [[ ! -f "$ROOT_DIR/$EMRDM_CKPT_FPATH" ]]; then
  echo "[error] Missing checkpoint: $EMRDM_CKPT_FPATH"
  exit 1
fi

# Align dataset keys with pair keys to avoid KeyError during EMRDM lookup.
EFFECTIVE_DATASET_FPATH="$DATASET_FPATH"
EFFECTIVE_DATASET_FPATH="$(python3 - "$ROOT_DIR" "$DATASET_FPATH" "$EMRDM_PAIRS_FPATH" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
dataset_rel = Path(sys.argv[2])
pairs_rel = Path(sys.argv[3])
dataset_path = root / dataset_rel
pairs_path = root / pairs_rel

with dataset_path.open() as f:
  dataset = json.load(f)
with pairs_path.open() as f:
  pairs = json.load(f)

missing = [k for k in dataset if k not in pairs]
if not missing:
  print(dataset_rel.as_posix())
  raise SystemExit(0)

aligned = {k: v for k, v in dataset.items() if k in pairs}
out_rel = Path("setup/.emrdm_samples_aligned.json")
out_path = root / out_rel
with out_path.open("w") as f:
  json.dump(aligned, f, indent=2)

print(f"[warn] {len(missing)} sample ids have no pair pointer; writing aligned dataset: {out_rel.as_posix()}", file=sys.stderr)
for k in missing[:10]:
  print(f"[warn] missing pair: {k}", file=sys.stderr)
if len(missing) > 10:
  print(f"[warn] ... and {len(missing)-10} more", file=sys.stderr)

print(out_rel.as_posix())
PY
)"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "[build] $IMAGE"
  docker build -f "$ROOT_DIR/Dockerfile.emrdm" -t "$IMAGE" "$ROOT_DIR"
else
  echo "[reuse] $IMAGE"
fi

SELECTED_ROIS_ARGS=()
if [[ -n "${SELECTED_ROIS:-}" ]]; then
  # shellcheck disable=SC2206
  ROIS=(${SELECTED_ROIS})
  SELECTED_ROIS_ARGS=(--selected-rois "${ROIS[@]}")
fi

echo "[run] EMRDM stratified eval"
echo "      dataset: $EFFECTIVE_DATASET_FPATH"
echo "      pairs:   $EMRDM_PAIRS_FPATH"

docker run --rm --gpus all \
  -v "$ROOT_DIR":/workspace \
  -w /workspace \
  "$IMAGE" \
  python3.10 benchmark.py \
    --dataset-fpath "$EFFECTIVE_DATASET_FPATH" \
    --model-name EMRDM \
    --device cuda \
    --batch-size "$BATCH_SIZE" \
    --aux-sensors s1 \
    --emrdm-config-fpath "$EMRDM_CONFIG_FPATH" \
    --emrdm-ckpt-fpath "$EMRDM_CKPT_FPATH" \
    --emrdm-pairs-fpath "$EMRDM_PAIRS_FPATH" \
    "${SELECTED_ROIS_ARGS[@]}"
