#!/home/pi/ultrasonic_gesture/venv/bin/python3
"""
Plot delay breakdown - Fixed version
"""
import matplotlib.pyplot as plt
import numpy as np

# Data
labels = ['Data Acquisition', 'Feature Extraction', 'SVM Inference', 'Game Logic', 'Median Filter']
sizes = [81.3, 6.4, 7.6, 3.3, 1.4]
colors = ['#4A90D9', '#F5A623', '#7ED321', '#D0021B', '#9B9B9B']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: Pie chart
explode = (0.05, 0.03, 0.03, 0.03, 0.03)
wedges, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
                                   autopct='%1.1f%%', shadow=True, startangle=90,
                                   textprops={'fontsize': 11, 'fontweight': 'bold'})
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(11)
    autotext.set_fontweight('bold')
ax1.set_title('End-to-End Latency Breakdown', fontsize=16, fontweight='bold', pad=15)

# Right: Horizontal bar chart - FIXED
y_pos = np.arange(len(labels))
bars = ax2.barh(y_pos, sizes, color=colors, edgecolor='white', linewidth=2, height=0.6)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(labels, fontsize=11)
ax2.set_xlabel('Latency Percentage (%)', fontsize=12, fontweight='bold')
ax2.set_title('Latency Distribution by Component', fontsize=16, fontweight='bold', pad=15)

# 在条形右侧显示数值 - 调整位置避免重叠
for i, (bar, v) in enumerate(zip(bars, sizes)):
    ax2.text(v + 1.5, i, f'{v:.1f}%', va='center', fontsize=12, fontweight='bold')

ax2.set_xlim(0, 100)
ax2.grid(axis='x', alpha=0.3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('experiment_results/delay_breakdown_pie.png', dpi=300, bbox_inches='tight')
print("✅ Saved: experiment_results/delay_breakdown_pie.png")
