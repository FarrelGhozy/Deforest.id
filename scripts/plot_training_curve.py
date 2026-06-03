import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

epochs_known = np.array([1, 5, 9, 10, 15, 20, 25, 30, 37, 42, 50])
train_loss = np.array([0.1974, 0.1775, 0.1759, 0.1753, 0.1704, 0.1670, 0.1655, 0.1631, 0.1598, 0.1565, 0.1555])
val_loss = np.array([0.2753, 0.2773, 0.2765, 0.2836, 0.2792, 0.2814, 0.2928, 0.2877, 0.2859, 0.2819, 0.2865])
val_iou = np.array([0.3788, 0.3726, 0.3831, 0.3712, 0.3628, 0.3636, 0.3653, 0.3628, 0.3649, 0.3663, 0.3616])

epoch_fine = np.linspace(1, 50, 200)
train_loss_smooth = np.interp(epoch_fine, epochs_known, train_loss)
val_loss_smooth = np.interp(epoch_fine, epochs_known, val_loss)
val_iou_smooth = np.interp(epoch_fine, epochs_known, val_iou)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(epoch_fine, train_loss_smooth, 'b-', linewidth=1.5, alpha=0.7, label='_nolegend_')
ax1.plot(epoch_fine, val_loss_smooth, 'r-', linewidth=1.5, alpha=0.7, label='_nolegend_')
ax1.scatter(epochs_known, train_loss, color='blue', s=40, zorder=5, label='Train Loss')
ax1.scatter(epochs_known, val_loss, color='red', s=40, zorder=5, marker='s', label='Val Loss')
ax1.axvline(x=9, color='gray', linestyle='--', alpha=0.5, label='Best model (epoch 9)')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Training and Validation Loss', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(1, 50)

ax2.plot(epoch_fine, val_iou_smooth, 'g-', linewidth=1.5, alpha=0.7, label='Val IoU (interpolated)')
ax2.scatter(epochs_known, val_iou, color='green', s=60, zorder=5, label='Val IoU (logged)')
ax2.axvline(x=9, color='gray', linestyle='--', alpha=0.5, label='Best (epoch 9)')
ax2.scatter([9], [0.3831], color='darkgreen', s=120, zorder=6, marker='*')
ax2.annotate(f'Best IoU = 0.3831', xy=(9, 0.3831), xytext=(22, 0.382),
             arrowprops=dict(arrowstyle='->', color='darkgreen'), fontsize=10,
             color='darkgreen', fontweight='bold')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('IoU', fontsize=12)
ax2.set_title('Validation IoU (Deforest Class)', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(1, 50)
ax2.set_ylim(0.35, 0.39)

plt.tight_layout()
plt.savefig('paper/assets2/training_curve.png', dpi=200, bbox_inches='tight')
print("Saved paper/assets2/training_curve.png")

fig2, ax3 = plt.subplots(figsize=(8, 5))
metrics = ['IoU', 'Dice', 'Precision', 'Recall']
values = [0.7256, 0.8410, 0.8340, 0.8480]
colors_bar = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
bars = ax3.bar(metrics, values, color=colors_bar, width=0.5, edgecolor='black', linewidth=1.2)
for bar, val in zip(bars, values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax3.set_ylim(0, 1.0)
ax3.set_ylabel('Score', fontsize=12)
ax3.set_title('Test Set Performance Metrics', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('paper/assets2/test_metrics.png', dpi=200, bbox_inches='tight')
print("Saved paper/assets2/test_metrics.png")
