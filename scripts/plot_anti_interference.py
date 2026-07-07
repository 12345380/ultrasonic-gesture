#!/home/pi/ultrasonic_gesture/venv/bin/python3
"""
Plot anti-interference results - Fixed version
"""
import matplotlib.pyplot as plt
import numpy as np

# Data - 缩短标签
conditions = ['Baseline', 'Light\nVariation', 'Ultrasonic\nInterference', 'Airflow\nInterference']
offset = [0.00, 0.02, 0.01, -0.01]
std = [0.06, 0.22, 0.05, 0.03]
outlier = [0.0, 0.0, 0.0, 0.0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

x = np.arange(len(conditions))
width = 0.35

# Left: Offset and Std
bars1 = ax1.bar(x - width/2, offset, width, label='Distance Offset (cm)', 
                color='#4A90D9', edgecolor='black', linewidth=1.2, alpha=0.8)
bars2 = ax1.bar(x + width/2, std, width, label='Distance Std (cm)',
                color='#F5A623', edgecolor='black', linewidth=1.2, alpha=0.8)

ax1.set_xticks(x)
ax1.set_xticklabels(conditions, fontsize=10)
ax1.set_ylabel('Distance (cm)', fontsize=12, fontweight='bold')
ax1.set_title('Interference Test: Offset and Std Deviation', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11, loc='upper right')
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_ylim(-0.1, 0.35)

# 数值标签
for bar, val in zip(bars1, offset):
    y_pos = val + 0.02 if val >= 0 else val - 0.02
    va = 'bottom' if val >= 0 else 'top'
    ax1.annotate(f'{val:+.2f}', xy=(bar.get_x() + bar.get_width()/2, val),
                xytext=(0, 5 if val >= 0 else -5), textcoords="offset points",
                ha='center', va=va, fontsize=10, fontweight='bold')

for bar, val in zip(bars2, std):
    ax1.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, val),
                xytext=(0, 5), textcoords="offset points", 
                ha='center', va='bottom', fontsize=10, fontweight='bold')

# Right: Outlier ratio
bar3 = ax2.bar(x, outlier, color='#7ED321', edgecolor='black', linewidth=1.2, alpha=0.8)
ax2.set_xticks(x)
ax2.set_xticklabels(conditions, fontsize=10)
ax2.set_ylabel('Outlier Ratio (%)', fontsize=12, fontweight='bold')
ax2.set_title('Interference Test: Outlier Ratio', fontsize=14, fontweight='bold')
ax2.set_ylim(0, 5)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

for bar, val in zip(bar3, outlier):
    ax2.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, val),
                xytext=(0, 5), textcoords="offset points", 
                ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('experiment_results/anti_interference_results.png', dpi=300, bbox_inches='tight')
print("✅ Saved: experiment_results/anti_interference_results.png")
