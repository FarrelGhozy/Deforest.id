.PHONY: help verify train-gfw train-ndvi evaluate visualize clean

help:
	@echo "Deforest.id Pipeline"
	@echo ""
	@echo "  make verify         Check cloud mask consistency"
	@echo "  make train-gfw      Train U-Net with GFW labels + cloud mask"
	@echo "  make train-ndvi     Train U-Net with NDVI labels + cloud mask (ablation)"
	@echo "  make evaluate       Run inference on test set and save metrics"
	@echo "  make visualize      Generate comparison figures"
	@echo "  make clean          Remove old models and predictions"

# ── Paths ──────────────────────────────────────────────────────────
DATA_DIR    := data/training/unet
MANIFEST    := $(DATA_DIR)/manifest.json
MODELS_DIR  := models

# ── Verification ───────────────────────────────────────────────────
verify:
	python scripts/verify_cloud.py \
		--chips-dir $(DATA_DIR)/chips \
		--labels-dir $(DATA_DIR)/labels_gfw \
		--raw-dir $(DATA_DIR)/raw \
		--sample 200

# ── Training ───────────────────────────────────────────────────────
train-gfw:
	python scripts/train_unet.py \
		--manifest $(MANIFEST) \
		--output $(MODELS_DIR)/unet_deforest_v2 \
		--epochs 50 --batch-size 32 --lr 1e-3 \
		--label-key mask

train-ndvi:
	python scripts/train_unet.py \
		--manifest $(MANIFEST) \
		--output $(MODELS_DIR)/unet_ndvi \
		--epochs 50 --batch-size 32 --lr 1e-3 \
		--label-key label_ndvi

# ── Evaluation ─────────────────────────────────────────────────────
evaluate:
	python scripts/infer_unet.py \
		--manifest $(MANIFEST) \
		--model $(MODELS_DIR)/unet_deforest_v2/best.pth \
		--output $(DATA_DIR)/predictions

evaluate-ndvi:
	python scripts/infer_unet.py \
		--manifest $(MANIFEST) \
		--model $(MODELS_DIR)/unet_ndvi/best.pth \
		--output $(DATA_DIR)/predictions_ndvi

# ── Cleanup ────────────────────────────────────────────────────────
clean:
	rm -rf $(MODELS_DIR)/unet_deforest_v2
	rm -rf $(MODELS_DIR)/unet_ndvi
	rm -rf $(DATA_DIR)/predictions
	rm -rf $(DATA_DIR)/predictions_ndvi
