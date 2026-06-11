#!/bin/bash
BASE=/mnt/c/Users/LABTI/Deforest.id
cd $BASE/services/gee-export/src

for i in 2 3 4 5 6 7; do
  echo ""
  echo "==================== Sample $i ===================="
  PYTHONPATH=. \
  GEE_PROJECT=form-sembako-chain \
  GEE_CREDENTIALS=$BASE/config/form-sembako-sa-key.json \
  python3.12 -m run export \
    --geojson $BASE/data/sample_$i.geojson \
    --prefix hl_sample_$i
  echo "EXIT: $?"
done

echo ""
echo "===== SEMUA EXPORT SELESAI ====="
