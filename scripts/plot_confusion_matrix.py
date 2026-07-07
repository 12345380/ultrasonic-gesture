#!/home/pi/ultrasonic_gesture/venv/bin/python3
"""
Plot confusion matrix heatmap - Fixed version
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Confusion matrix data
cm = np.array([
    [15, 0, 0, 0],
    [0, 15, 0, 0],
    [0, 0, 15, 0],
    [0, 0, 0, 60]
])

labels = ['c1_Stable', 'c2_Push', 'c3_Pull', 'c4_Wave']

colors = ['#ffffff', '#4A90D9', '#1a3a5c']
cmap = LinearSegmentedColormap.from_list('custom_blue', colors, N=256)

fig, ax = plt.subplots(figsize=(8, 7))

im = ax.imshow(cm, cmap=cmap, interpolation='nearest')

ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels, fontsize=13, fontweight='bold')
ax.set_yticklabels(labels, fontsize=13, fontweight='bold')

# Color bar
cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
cbar.ax.set_ylabel('Sample Count', rotation=-90, va="bottom", fontsize=12, fontweight='bold')

# 显示数值 - 优化颜色
for i in range(len(labels)):
    for j in range(len(labels)):
        val = cm[i, j]
        color = 'white' if val > 30 else 'black'
        ax.text(j, i, val, ha="center", va="center", 
                color=color, fontsize=20, fontweight='bold')

ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
ax.set_title('SVM Confusion Matrix (Test Set: 105 Samples)', fontsize=15, fontweight='bold', pad=15)

# Grid lines
ax.set_xticks(np.arange(-.5, len(labels), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(labels), 1), minor=True)
ax.grid(which="minor", color="black", linestyle='-', linewidth=1.5)

plt.tight_layout()
plt.savefig('experiment_results/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✅ Saved: experiment_results/confusion_matrix.png")
