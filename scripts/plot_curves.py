import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('models/deforest_multitemporal/train_log.csv')
fig, axes = plt.subplots(2, 2, figsize=(10, 7))

axes[0,0].plot(df['epoch'], df['IoU'], 'o-', color='#2c7bb6')
axes[0,0].set_xlabel('Epoch'); axes[0,0].set_ylabel('IoU')
axes[0,0].set_title('Validation IoU')
axes[0,0].axvline(5, color='red', ls='--', alpha=0.5, label='best epoch=5')
axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

axes[0,1].plot(df['epoch'], df['Precision'], 's-', color='#d7191c', label='Precision')
axes[0,1].plot(df['epoch'], df['Recall'], 's-', color='#1a9641', label='Recall')
axes[0,1].plot(df['epoch'], df['F1'], 's-', color='#fdae61', label='F1')
axes[0,1].set_xlabel('Epoch'); axes[0,1].set_ylabel('Metric')
axes[0,1].set_title('Precision / Recall / F1')
axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)

axes[1,0].plot(df['epoch'], df['train_loss'], label='Train')
axes[1,0].plot(df['epoch'], df['val_loss'], label='Val')
axes[1,0].set_xlabel('Epoch'); axes[1,0].set_ylabel('Loss')
axes[1,0].set_title('Training & Validation Loss')
axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)

axes[1,1].plot(df['epoch'], df['lr']*1e5, 'o-', color='purple')
axes[1,1].set_xlabel('Epoch'); axes[1,1].set_ylabel('LR x 1e5')
axes[1,1].set_title('Learning Rate')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('reports/training_curves.png', dpi=200, bbox_inches='tight')
print('Saved reports/training_curves.png')
