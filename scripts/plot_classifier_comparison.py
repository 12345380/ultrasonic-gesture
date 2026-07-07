#!/home/pi/ultrasonic_gesture/venv/bin/python3
"""
Plot classifier comparison - Fixed version
"""
import matplotlib.pyplot as plt
import numpy as np

# Data
classifiers = ['Threshold\n(Baseline)', 'SVM\n(RBF Kernel)', '1D-CNN\n(MLP)']
metrics = ['Accuracy (%)', 'Macro Precision', 'Macro Recall', 'Macro F1']
data = {
    'Accuracy (%)': [71.4, 100.0, 91.4],
    'Macro Precision': [0.417, 1.000, 0.906],
    'Macro Recall': [0.500, 1.000, 0.963],
    'Macro F1': [0.450, 1.000, 0.922]
}
colors = ['#E8A87C', '#85CDCA', '#6C5B7B']

x = np.arange(len(classifiers))
width = 0.2
fig, ax = plt.subplots(figsize=(12, 7))

# Draw grouped bars
for i, (metric, values) in enumerate(data.items()):
    offset = (i - 1.5) * width
    bars = ax.bar(x + offset, values, width, label=metric, 
                  color=colors[i % len(colors)], 
                  edgecolor='black', linewidth=0.8, alpha=0.85)
    # 数值标签 - 调整位置避免重叠
    for bar, val in zip(bars, values):
        if metric == 'Accuracy (%)':
            label = f'{val:.1f}%'
        else:
            label = f'{val:.3f}'
        # 数值显示在柱子内部顶部
        y_pos = val + 0.02 if val < 1.0 else val + 1.0
        ax.annotate(label, xy=(bar.get_x() + bar.get_width()/2, val),
                   xytext=(0, 5), textcoords="offset points", 
                   ha='center', va='bottom',
                   fontsize=9, fontweight='bold', rotation=0)

ax.set_xlabel('Classifier', fontsize=13, fontweight='bold')
ax.set_ylabel('Score', fontsize=13, fontweight='bold')
ax.set_title('Performance Comparison of Three Classifiers (4-class Gesture Classification)', 
             fontsize=15, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(classifiers, fontsize=12)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=4, fontsize=11)
ax.set_ylim(0, 1.2)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('experiment_results/classifier_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: experiment_results/classifier_comparison.png")
