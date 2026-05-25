#!/usr/bin/env bash
set -euo pipefail

# Run from repository root:
# bash run_on_vm.sh

python -m src.prepare_crops \
  --data-root /root/.cache/kagglehub/datasets/abdelrahmanelgharibx/coco2017-subset/versions/1 \
  --num-classes 4 \
  --min-train-objects 500 \
  --max-train-objects 2000 \
  --min-val-objects 5 \
  --max-train-per-class 300 \
  --max-val-per-class 80

python -m src.generate_controlnet \
  --images-per-class 40 \
  --num-inference-steps 20 \
  --image-size 512

python -m src.train_classifier \
  --experiment both \
  --epochs 5 \
  --batch-size 32 \
  --image-size 224
