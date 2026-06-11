#!/bin/bash
cd /mnt/c/Users/LABTI/Deforest.id || exit 1
mkdir -p models/unet_deforest_v2
export PYTHONPATH=/usr/lib/python3/dist-packages:$PYTHONPATH
exec /mnt/c/Users/LABTI/Deforest.id/.venv/bin/python \
  scripts/train_unet.py \
  --manifest data/training/unet/manifest.json \
  --output models/unet_deforest_v2 \
  --epochs 50 --batch-size 64 --lr 1e-3 \
  --label-key mask \
  --cache-ram \
  > models/unet_deforest_v2/train.log 2>&1
