#!/home/pi/ultrasonic_gesture/venv/bin/python3
"""
实验四：系统抗干扰测试 (RQ4)
测试系统在常见环境干扰下的稳定性
"""
import RPi.GPIO as GPIO
import time
import numpy as np
import json
import os
from datetime import datetime

# 引脚定义
TRIG = 23
ECHO = 24

class AntiInterferenceTest:
    def __init__(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(TRIG, GPIO.OUT)
        GPIO.setup(ECHO, GPIO.IN)
        
        # 测试参数
        self.test_duration = 60  # 每次测试60秒
        self.sample_interval = 0.02  # 50Hz
        
        # 干扰类型
        self.interference_types = {
            'baseline': '无干扰基线',
            'light': '环境光变化',
            'sound': '同频声波干扰',
            'airflow': '气流干扰'
        }
        
        # 存储结果
        self.results = {}
        self.baseline_mean = None  # 存储基线平均值
        
        # 创建结果目录
        self.result_dir = 'experiment_results'
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
    
    def get_distance(self):
        """获取单次距离测量"""
        GPIO.output(TRIG, False)
        time.sleep(0.005)
        
        GPIO.output(TRIG, True)
        time.sleep(0.00001)
        GPIO.output(TRIG, False)
        
        pulse_start = time.time()
        timeout = pulse_start + 0.04
        
        while GPIO.input(ECHO) == 0 and pulse_start < timeout:
            pulse_start = time.time()
        
        if pulse_start >= timeout:
            return None
        
        pulse_end = time.time()
        while GPIO.input(ECHO) == 1 and pulse_end < timeout:
            pulse_end = time.time()
        
        if pulse_end >= timeout:
            return None
        
        pulse_duration = pulse_end - pulse_start
        distance = pulse_duration * 17150
        
        if 2 <= distance <= 400:
            return distance
        return None
    
    def collect_data(self, duration, description):
        """采集一段时间的数据"""
        print(f"\n   采集数据中... (持续{duration}秒)")
        
        distances = []
        timestamps = []
        start_time = time.time()
        
        # 显示进度条
        progress_width = 40
        
        while time.time() - start_time < duration:
            dist = self.get_distance()
            if dist is not None:
                distances.append(dist)
                timestamps.append(time.time() - start_time)
            
            # 显示进度
            elapsed = time.time() - start_time
            progress = int(elapsed / duration * progress_width)
            bar = "█" * progress + "░" * (progress_width - progress)
            print(f"\r   [{bar}] {elapsed:.1f}/{duration}s | 样本: {len(distances)}", end='')
            
            time.sleep(self.sample_interval)
        
        print()  # 换行
        print(f"   采集完成，共 {len(distances)} 个样本")
        
        return distances, timestamps
    
    def test_baseline(self):
        """测试基线：无干扰"""
        print("\n" + "-"*60)
        print("🔵 测试基线: 无干扰")
        print("-"*60)
        print("请将手放在传感器前20-30cm处，保持稳定")
        input("准备好后按回车开始...")
        
        distances, timestamps = self.collect_data(self.test_duration, "基线")
        
        return self.calculate_stats(distances, None)
    
    def test_light_interference(self):
        """测试干扰1：环境光变化"""
        print("\n" + "-"*60)
        print("🟡 测试干扰1: 环境光变化")
        print("-"*60)
        print("请在传感器旁边打开/关闭灯光（如台灯）")
        print("模拟室内光照波动")
        print("请保持手在传感器前20-30cm处")
        input("准备好后按回车开始...")
        
        print("\n   请开始快速开关灯光（每3-5秒切换一次）...")
        
        distances, timestamps = self.collect_data(self.test_duration, "环境光干扰")
        
        return self.calculate_stats(distances, self.baseline_mean)
    
    def test_sound_interference(self):
        """测试干扰2：同频声波干扰"""
        print("\n" + "-"*60)
        print("🔴 测试干扰2: 同频声波干扰")
        print("-"*60)
        print("⚠️  这个测试需要第二个HC-SR04传感器")
        print("   将第二个传感器放在当前传感器旁边30cm处")
        print("   第二个传感器会随机发射超声波")
        print("   如果没有第二个传感器，可以用手在传感器前快速晃动模拟")
        print("请保持手在传感器前20-30cm处")
        input("准备好后按回车开始...")
        
        print("\n   请启动干扰源...")
        
        distances, timestamps = self.collect_data(self.test_duration, "声波干扰")
        
        return self.calculate_stats(distances, self.baseline_mean)
    
    def test_airflow_interference(self):
        """测试干扰3：气流干扰"""
        print("\n" + "-"*60)
        print("🟢 测试干扰3: 气流干扰")
        print("-"*60)
        print("请在传感器和手之间打开风扇（低速档）")
        print("请保持手在传感器前20-30cm处")
        input("准备好后按回车开始...")
        
        print("\n   请打开风扇（低速档）...")
        
        distances, timestamps = self.collect_data(self.test_duration, "气流干扰")
        
        return self.calculate_stats(distances, self.baseline_mean)
    
    def calculate_stats(self, distances, reference_mean):
        """计算统计数据"""
        if len(distances) == 0:
            return None
        
        arr = np.array(distances)
        mean = np.mean(arr)
        std = np.std(arr)
        min_val = np.min(arr)
        max_val = np.max(arr)
        range_val = max_val - min_val
        
        # 计算异常值比例（偏移>5cm）
        if reference_mean is not None:
            outliers = np.sum(np.abs(arr - reference_mean) > 5) / len(arr) * 100
            offset = mean - reference_mean
        else:
            outliers = 0
            offset = 0
        
        return {
            'mean': mean,
            'std': std,
            'min': min_val,
            'max': max_val,
            'range': range_val,
            'outlier_percent': outliers,
            'offset': offset,
            'n_samples': len(distances)
        }
    
    def run_all_tests(self):
        """运行所有抗干扰测试"""
        print("="*70)
        print("实验四：系统抗干扰测试 (RQ4)")
        print("="*70)
        print("\n实验说明:")
        print("本实验测试系统在以下干扰下的稳定性:")
        print("  1. 环境光变化（灯光开关）")
        print("  2. 同频声波干扰（第二个超声波传感器或手晃动模拟）")
        print("  3. 气流干扰（风扇吹风）")
        print("\n每次测试持续60秒，采集约3000个数据点")
        print("-"*70)
        
        # 1. 基线测试
        print("\n📊 开始测试...")
        baseline_stats = self.test_baseline()
        if baseline_stats:
            self.baseline_mean = baseline_stats['mean']
            self.results['baseline'] = baseline_stats
            print(f"\n   基线结果:")
            print(f"     均值: {baseline_stats['mean']:.2f} cm")
            print(f"     标准差: {baseline_stats['std']:.2f} cm")
            print(f"     范围: {baseline_stats['min']:.1f} - {baseline_stats['max']:.1f} cm")
        
        # 暂停，让用户准备下一个测试
        input("\n按回车继续下一个测试...")
        
        # 2. 环境光变化
        light_stats = self.test_light_interference()
        if light_stats:
            self.results['light'] = light_stats
            print(f"\n   环境光干扰结果:")
            print(f"     均值: {light_stats['mean']:.2f} cm")
            print(f"     标准差: {light_stats['std']:.2f} cm")
            print(f"     偏移: {light_stats['offset']:+.2f} cm")
            print(f"     异常率: {light_stats['outlier_percent']:.1f}%")
        
        input("\n按回车继续下一个测试...")
        
        # 3. 同频声波干扰
        sound_stats = self.test_sound_interference()
        if sound_stats:
            self.results['sound'] = sound_stats
            print(f"\n   声波干扰结果:")
            print(f"     均值: {sound_stats['mean']:.2f} cm")
            print(f"     标准差: {sound_stats['std']:.2f} cm")
            print(f"     偏移: {sound_stats['offset']:+.2f} cm")
            print(f"     异常率: {sound_stats['outlier_percent']:.1f}%")
        
        input("\n按回车继续下一个测试...")
        
        # 4. 气流干扰
        airflow_stats = self.test_airflow_interference()
        if airflow_stats:
            self.results['airflow'] = airflow_stats
            print(f"\n   气流干扰结果:")
            print(f"     均值: {airflow_stats['mean']:.2f} cm")
            print(f"     标准差: {airflow_stats['std']:.2f} cm")
            print(f"     偏移: {airflow_stats['offset']:+.2f} cm")
            print(f"     异常率: {airflow_stats['outlier_percent']:.1f}%")
        
        # 显示汇总结果
        self.print_summary()
        self.save_results()
    
    def print_summary(self):
        """打印汇总结果（论文格式）"""
        print("\n" + "="*70)
        print("表4 系统抗干扰测试结果")
        print("="*70)
        print(f"{'条件':<20} {'距离均值偏移(cm)':<20} {'距离标准差(cm)':<18} {'异常值(>5cm偏移)比例':<20}")
        print("-"*70)
        
        # 基线
        baseline = self.results.get('baseline')
        if baseline:
            print(f"{'无干扰基线':<20} {'0.00':<20} {baseline['std']:.2f}{'':<16} {'0.0%':<20}")
        
        # 环境光变化
        light = self.results.get('light')
        if light:
            offset_str = f"{light['offset']:+.2f}" if 'offset' in light else "N/A"
            outlier_str = f"{light['outlier_percent']:.1f}%" if 'outlier_percent' in light else "N/A"
            print(f"{'环境光变化':<20} {offset_str:<20} {light['std']:.2f}{'':<16} {outlier_str:<20}")
        
        # 同频声波干扰
        sound = self.results.get('sound')
        if sound:
            offset_str = f"{sound['offset']:+.2f}" if 'offset' in sound else "N/A"
            outlier_str = f"{sound['outlier_percent']:.1f}%" if 'outlier_percent' in sound else "N/A"
            print(f"{'同频声波干扰':<20} {offset_str:<20} {sound['std']:.2f}{'':<16} {outlier_str:<20}")
        
        # 气流干扰
        airflow = self.results.get('airflow')
        if airflow:
            offset_str = f"{airflow['offset']:+.2f}" if 'offset' in airflow else "N/A"
            outlier_str = f"{airflow['outlier_percent']:.1f}%" if 'outlier_percent' in airflow else "N/A"
            print(f"{'气流干扰':<20} {offset_str:<20} {airflow['std']:.2f}{'':<16} {outlier_str:<20}")
        
        print("="*70)
    
    def save_results(self):
        """保存实验结果"""
        # 保存JSON
        json_path = os.path.join(self.result_dir, 'anti_interference_results.json')
        
        json_data = {}
        for key, stats in self.results.items():
            if stats:
                json_data[key] = {
                    'mean': stats['mean'],
                    'std': stats['std'],
                    'min': stats['min'],
                    'max': stats['max'],
                    'range': stats['range'],
                    'outlier_percent': stats.get('outlier_percent', 0),
                    'offset': stats.get('offset', 0),
                    'n_samples': stats['n_samples']
                }
        
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"\n✅ 结果已保存到: {json_path}")
        
        # 保存表格格式
        txt_path = os.path.join(self.result_dir, 'anti_interference_table.txt')
        with open(txt_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("表4 系统抗干扰测试结果\n")
            f.write("="*70 + "\n")
            f.write(f"{'条件':<20} {'距离均值偏移(cm)':<20} {'距离标准差(cm)':<18} {'异常值(>5cm偏移)比例':<20}\n")
            f.write("-"*70 + "\n")
            
            baseline = self.results.get('baseline')
            if baseline:
                f.write(f"{'无干扰基线':<20} {'0.00':<20} {baseline['std']:.2f}{'':<16} {'0.0%':<20}\n")
            
            light = self.results.get('light')
            if light:
                offset_str = f"{light['offset']:+.2f}" if 'offset' in light else "N/A"
                outlier_str = f"{light['outlier_percent']:.1f}%" if 'outlier_percent' in light else "N/A"
                f.write(f"{'环境光变化':<20} {offset_str:<20} {light['std']:.2f}{'':<16} {outlier_str:<20}\n")
            
            sound = self.results.get('sound')
            if sound:
                offset_str = f"{sound['offset']:+.2f}" if 'offset' in sound else "N/A"
                outlier_str = f"{sound['outlier_percent']:.1f}%" if 'outlier_percent' in sound else "N/A"
                f.write(f"{'同频声波干扰':<20} {offset_str:<20} {sound['std']:.2f}{'':<16} {outlier_str:<20}\n")
            
            airflow = self.results.get('airflow')
            if airflow:
                offset_str = f"{airflow['offset']:+.2f}" if 'offset' in airflow else "N/A"
                outlier_str = f"{airflow['outlier_percent']:.1f}%" if 'outlier_percent' in airflow else "N/A"
                f.write(f"{'气流干扰':<20} {offset_str:<20} {airflow['std']:.2f}{'':<16} {outlier_str:<20}\n")
            
            f.write("="*70 + "\n")
        
        print(f"✅ 表格已保存到: {txt_path}")
    
    def cleanup(self):
        GPIO.cleanup()

def main():
    tester = AntiInterferenceTest()
    try:
        tester.run_all_tests()
    finally:
        tester.cleanup()
        print("\n✅ GPIO已清理")

if __name__ == "__main__":
    main()
