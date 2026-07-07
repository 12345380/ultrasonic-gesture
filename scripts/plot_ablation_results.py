#!/home/pi/ultrasonic_gesture/venv/bin/python3
"""
Plot ablation study results - Fixed version
"""
import matplotlib.pyplot as plt
import numpy as np

# Data - 缩短标签避免重叠
configs = ['All Features\n(11 dims)', 'Remove F9\n(Jitter)', 'Remove F10\n(Consistency)', 
           'Remove F11\n(Smoothness)', 'Remove All\nDomain Feat.', 'General\nOnly']
accuracy = [94.29, 90.00, 94.29, 97.14, 90.00, 90.00]
f1 = [0.9427, 0.8998, 0.9427, 0.9714, 0.8998, 0.8998]

colors_acc = ['#4A90D9', '#D0021B', '#4A90D9', '#7ED321', '#D0021B', '#D0021B']
colors_f1 = ['#4A90D9', '#D0021B', '#4A90D9', '#7ED321', '#D0021B', '#D0021B']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

x = np.arange(len(configs))

# Left: Accuracy
bars1 = ax1.bar(x, accuracy, color=colors_acc, edgecolor='black', linewidth=1.2, alpha=0.8)
ax1.set_xticks(x)
ax1.set_xticklabels(configs, fontsize=9, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Classification Accuracy with Different Feature Sets', fontsize=14, fontweight='bold')
ax1.set_ylim(85, 100)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

for bar, val in zip(bars1, accuracy):
    ax1.annotate(f'{val:.2f}%', xy=(bar.get_x() + bar.get_width()/2, val),
                xytext=(0, 5), textcoords="offset points", 
                ha='center', va='bottom',
                fontsize=10, fontweight='bold')

ax1.axhline(y=94.29, color='#4A90D9', linestyle='--', linewidth=2, alpha=0.6)

# Right: Macro F1
bars2 = ax2.bar(x, f1, color=colors_f1, edgecolor='black', linewidth=1.2, alpha=0.8)
ax2.set_xticks(x)
ax2.set_xticklabels(configs, fontsize=9, fontweight='bold')
ax2.set_ylabel('Macro F1-Score', fontsize=12, fontweight='bold')
ax2.set_title('Macro F1-Score with Different Feature Sets', fontsize=14, fontweight='bold')
ax2.set_ylim(0.85, 1.0)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

for bar, val in zip(bars2, f1):
    ax2.annotate(f'{val:.4f}', xy=(bar.get_x() + bar.get_width()/2, val),
                xytext=(0, 5), textcoords="offset points", 
                ha='center', va='bottom',
                fontsize=10, fontweight='bold')

ax2.axhline(y=0.9427, color='#4A90D9', linestyle='--', linewidth=2, alpha=0.6)

plt.tight_layout()
plt.savefig('experiment_results/ablation_results.png', dpi=300, bbox_inches='tight')
print("✅ Saved: experiment_results/ablation_results.png")
