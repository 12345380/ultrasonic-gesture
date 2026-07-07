#!/home/pi/ultrasonic_gesture/venv/bin/python3
"""
Run all plotting scripts
"""
import subprocess
import os

os.makedirs('experiment_results', exist_ok=True)

print("="*60)
print("Generating Paper Figures")
print("="*60)

scripts = [
    ('plot_delay_pie.py', 'Delay Breakdown'),
    ('plot_classifier_comparison.py', 'Classifier Comparison'),
    ('plot_ablation_results.py', 'Ablation Results'),
    ('plot_anti_interference.py', 'Anti-Interference'),
    ('plot_confusion_matrix.py', 'Confusion Matrix')
]

for script, name in scripts:
    print(f"\n📊 Generating: {name}")
    try:
        result = subprocess.run(['python', f'scripts/{script}'], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✅ {name} - Success")
        else:
            print(f"   ❌ {name} - Error: {result.stderr}")
    except Exception as e:
        print(f"   ❌ {name} - Exception: {e}")

print("\n" + "="*60)
print("✅ All figures generated in experiment_results/")
print("="*60)
