#!/bin/bash
BASE=/mnt/c/Users/LABTI/Deforest.id
cd $BASE/services/gee-export/src
PYTHONPATH=. \
GEE_PROJECT=form-sembako-chain \
GEE_CREDENTIALS=$BASE/config/form-sembako-sa-key.json \
python3.12 -m run export \
  --geojson $BASE/data/sample_1.geojson \
  --prefix hl_sample_1
echo "EXIT: $?"
