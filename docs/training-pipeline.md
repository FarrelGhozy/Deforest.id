# Training Pipeline

U-Net training pipeline untuk segmentasi deforestasi.

---

## Arsitektur Model

```mermaid
flowchart LR
    A[Input: 5-channel<br/>RGB + NIR + NDVI] --> B[Encoder<br/>Down 4x]
    B --> C[Bottleneck]
    C --> D[Decoder<br/>Up 4x + Skip Connections]
    D --> E[Output Mask<br/>64x64]
```

| Komponen | Detail |
|----------|--------|
| Input | 5 channel: RGB (3) + NIR (1) + NDVI (1) |
| Encoder | 4 stage down-sampling, base filters=32 |
| Bottleneck | 512 filters |
| Decoder | 4 stage up-sampling + skip connections |
| Output | 2 channel (forest / deforest) |
| Total params | ~7.5M |

## Quick Start

Dataset harus sudah siap (chips + masks). Lihat [Annotation Pipeline](annotation-pipeline.md).

```bash
# Install dependencies
uv pip install -r scripts/requirements-training.txt

# Train
uv run python scripts/train_unet.py \
  --manifest data/training/unet/manifest.json \
  --output models/unet_deforest \
  --epochs 50 \
  --batch-size 32 \
  --lr 1e-3
```

## Training Details

### Loss Function

**Dice Loss** — lebih cocok untuk imbalanced segmentation (deforestasi biasanya <10% area):

```python
class DiceLoss(nn.Module):
    def forward(self, logits, targets):
        # Ignore class 255 (uncertain regions)
        # Dice = 1 - (2*|A∩B| + smooth) / (|A| + |B| + smooth)
```

### Metrics

| Metrik | Target | Keterangan |
|--------|--------|------------|
| Dice Coefficient | >0.85 | Per-class dice |
| IoU (deforest) | >0.70 | Intersection over Union untuk class deforestasi |
| Validation Loss | <0.15 | Dice loss |

### Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| Epochs | 50 | Early stopping jika val loss plateau >10 epoch |
| Batch size | 32 | Turunkan jika OOM |
| Learning rate | 1e-3 | Cosine annealing scheduler |
| Optimizer | AdamW | weight decay=1e-4 |
| Input size | 64×64 | Sesuai ukuran chip |

### Data Augmentation

- Horizontal flip (50%)
- Vertical flip (50%)
- Rotation ±15°
- Brightness/contrast jitter

## Monitoring

Training log:

```
Epoch   1/50 | Train Loss: 0.4521 | Val Loss: 0.3842 | Val IoU: 0.1234
Epoch  10/50 | Train Loss: 0.1823 | Val Loss: 0.1642 | Val IoU: 0.4567
Epoch  25/50 | Train Loss: 0.0956 | Val Loss: 0.0987 | Val IoU: 0.6789
Epoch  50/50 | Train Loss: 0.0543 | Val Loss: 0.0682 | Val IoU: 0.8123
```

Best model (by val IoU) disimpan otomatis ke `models/unet_deforest/best.pth`.

## Inference

Setelah training, model siap dipakai di pipeline inference:

```python
import torch
from train_unet import SimpleUNet

model = SimpleUNet(in_channels=5, out_channels=2)
model.load_state_dict(torch.load("models/unet_deforest/best.pth"))
model.eval()

# Input: (B, 5, 64, 64) — RGB + NIR + NDVI
with torch.no_grad():
    logits = model(chip_batch)
    mask = logits.argmax(dim=1)  # 0=forest, 1=deforest
```
