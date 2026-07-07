#!/home/pi/ultrasonic_gesture/venv/bin/python3
"""
实验一：测距精度测试 (RQ1) - 交互式版本
每次只测一个距离，显示数据质量评估，决定是否保存
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

class DistanceAccuracyTest:
    def __init__(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(TRIG, GPIO.OUT)
        GPIO.setup(ECHO, GPIO.IN)
        
        # 测试参数
        self.samples_per_point = 100  # 每个点采集100次
        self.sample_interval = 0.02   # 50Hz采样间隔
        
        # 测试点列表
        self.test_points = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
        
        # 存储结果
        self.results = []
        
        # 创建结果目录
        self.result_dir = 'experiment_results'
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        
        # 加载已有结果（如果存在）
        self.load_existing_results()
    
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
    
    def load_existing_results(self):
        """加载已保存的结果"""
        json_path = os.path.join(self.result_dir, 'distance_accuracy_results.json')
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                self.results = data
                print(f"📂 已加载 {len(self.results)} 个已保存的测试点")
            except:
                self.results = []
        else:
            self.results = []
    
    def save_results(self):
        """保存所有结果"""
        json_path = os.path.join(self.result_dir, 'distance_accuracy_results.json')
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # 同时保存表格格式
        self.save_table()
        print(f"\n✅ 结果已保存到: {json_path}")
    
    def save_table(self):
        """保存表格格式"""
        if not self.results:
            return
        
        txt_path = os.path.join(self.result_dir, 'distance_accuracy_table.txt')
        with open(txt_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("表1 HC-SR04测距精度测试结果\n")
            f.write("="*80 + "\n\n")
            f.write(f"{'真值距离(cm)':<15} {'测量均值(cm)':<15} {'标准差(cm)':<12} {'MAE(cm)':<12} {'误差百分比':<12}\n")
            f.write("-"*80 + "\n")
            
            mae_list = []
            std_list = []
            
            for r in sorted(self.results, key=lambda x: x['true_distance']):
                f.write(f"{r['true_distance']:<15} {r['mean']:<15.2f} {r['std']:<12.2f} {r['mae']:<12.2f} {r['error_percent']:<11.2f}%\n")
                mae_list.append(r['mae'])
                std_list.append(r['std'])
            
            f.write("-"*80 + "\n")
            f.write(f"最大MAE: {max(mae_list):.2f} cm\n")
            f.write(f"平均MAE: {np.mean(mae_list):.2f} cm\n")
            f.write(f"最大标准差: {max(std_list):.2f} cm\n")
        
        print(f"✅ 表格已保存到: {txt_path}")
    
    def show_progress(self):
        """显示当前进度"""
        completed = [r['true_distance'] for r in self.results]
        remaining = [d for d in self.test_points if d not in completed]
        
        print(f"\n📊 进度: {len(completed)}/{len(self.test_points)} 个测试点已完成")
        if remaining:
            print(f"   待测试: {remaining}")
        else:
            print("   ✅ 所有测试点已完成！")
    
    def show_instruction(self, target_distance):
        """显示测试说明"""
        print("\n" + "="*60)
        print(f"📏 测试点: {target_distance} cm")
        print("="*60)
        print(f"\n操作说明:")
        print(f"  1. 将反射面放在距离传感器 {target_distance} cm 处")
        print(f"  2. 用尺子精确测量真实距离")
        print(f"  3. 确保反射面正对传感器，不要倾斜")
        print(f"  4. 系统将采集 {self.samples_per_point} 次读数")
        print(f"  5. 采集完成后显示数据质量评估")
        print(f"  6. 你可以选择保存或重新测量")
        print("-"*60)
    
    def collect_data(self, target_distance):
        """采集指定距离的数据"""
        print(f"\n📡 采集数据中... (共{self.samples_per_point}次)")
        print("   进度: [", end='')
        
        measurements = []
        progress_width = 50
        
        for i in range(self.samples_per_point):
            dist = self.get_distance()
            if dist is not None:
                measurements.append(dist)
            
            # 更新进度条
            progress = int((i + 1) / self.samples_per_point * progress_width)
            bar = "█" * progress + "░" * (progress_width - progress)
            print(f"\r   进度: [{bar}] {i+1}/{self.samples_per_point}", end='')
            
            time.sleep(self.sample_interval)
        
        print()  # 换行
        
        if len(measurements) < 10:
            print(f"   ⚠️  有效读数太少 ({len(measurements)}次)，请检查传感器连接")
            return None
        
        return measurements
    
    def evaluate_data(self, measurements, target_distance):
        """评估数据质量"""
        arr = np.array(measurements)
        mean = np.mean(arr)
        std = np.std(arr)
        mae = np.mean(np.abs(arr - target_distance))
        error_percent = (mae / target_distance) * 100 if target_distance > 0 else 0
        min_val = np.min(arr)
        max_val = np.max(arr)
        range_val = max_val - min_val
        
        # 质量评估
        issues = []
        warnings = []
        
        # 1. 检查MAE是否在合理范围内
        if target_distance <= 30:
            if mae > 1.0:
                issues.append(f"MAE={mae:.2f}cm 偏大 (应<1.0cm)")
            elif mae > 0.5:
                warnings.append(f"MAE={mae:.2f}cm 略大 (应<0.5cm)")
        elif target_distance <= 50:
            if mae > 2.0:
                issues.append(f"MAE={mae:.2f}cm 偏大 (应<2.0cm)")
            elif mae > 1.0:
                warnings.append(f"MAE={mae:.2f}cm 略大 (应<1.0cm)")
        else:  # >50cm
            if mae > 3.0:
                issues.append(f"MAE={mae:.2f}cm 偏大 (应<3.0cm)")
            elif mae > 1.5:
                warnings.append(f"MAE={mae:.2f}cm 略大 (应<1.5cm)")
        
        # 2. 检查标准差（稳定性）
        if std > 1.0:
            issues.append(f"标准差={std:.2f}cm 过大 (应<1.0cm)")
        elif std > 0.5:
            warnings.append(f"标准差={std:.2f}cm 略大 (应<0.5cm)")
        
        # 3. 检查偏差方向
        bias = mean - target_distance
        if abs(bias) > 2.0:
            issues.append(f"偏差={bias:+.2f}cm 过大")
        
        # 4. 检查有效读数比例
        valid_ratio = len(measurements) / self.samples_per_point
        if valid_ratio < 0.8:
            issues.append(f"有效读数比例={valid_ratio:.1%} 偏低")
        
        # 显示统计信息
        print(f"\n📊 统计结果:")
        print(f"   测量均值: {mean:.2f} cm")
        print(f"   标准差:   {std:.2f} cm")
        print(f"   最小值:   {min_val:.1f} cm")
        print(f"   最大值:   {max_val:.1f} cm")
        print(f"   变化范围: {range_val:.1f} cm")
        print(f"   MAE:      {mae:.2f} cm")
        print(f"   误差百分比: {error_percent:.2f}%")
        print(f"   有效读数: {len(measurements)}/{self.samples_per_point}")
        
        # 显示质量评估
        print(f"\n🔍 数据质量评估:")
        if issues:
            print(f"   ❌ 问题:")
            for issue in issues:
                print(f"      • {issue}")
        if warnings:
            print(f"   ⚠️  警告:")
            for warn in warnings:
                print(f"      • {warn}")
        if not issues and not warnings:
            print(f"   ✅ 数据质量良好！")
        
        return {
            'true_distance': target_distance,
            'mean': mean,
            'std': std,
            'mae': mae,
            'error_percent': error_percent,
            'min': min_val,
            'max': max_val,
            'range': range_val,
            'samples': len(measurements),
            'measurements': measurements,
            'issues': issues,
            'warnings': warnings,
            'is_good': len(issues) == 0
        }
    
    def get_user_decision(self, result):
        """询问用户是否保存"""
        print("\n" + "-"*60)
        
        if result['is_good']:
            print("✅ 数据质量良好，推荐保存")
            default = 'y'
        else:
            print("⚠️  数据存在一些问题，建议重新测量")
            default = 'n'
        
        choice = input(f"是否保存此数据？(y/n，默认{default}): ").strip().lower()
        if choice == '':
            choice = default
        
        return choice == 'y'
    
    def run_test(self):
        """运行测距精度测试"""
        print("="*60)
        print("📏 实验一：测距精度测试 (RQ1)")
        print("="*60)
        print("\n本测试将测量HC-SR04在不同距离的精度")
        print(f"测试点: {self.test_points} cm")
        print(f"每个点采集: {self.samples_per_point} 次")
        print("-"*60)
        
        # 显示进度
        self.show_progress()
        
        # 测试每个点
        for target_dist in self.test_points:
            # 检查是否已完成
            if any(r['true_distance'] == target_dist for r in self.results):
                print(f"\n⏭️  {target_dist}cm 已测试，跳过")
                continue
            
            # 显示说明
            self.show_instruction(target_dist)
            
            # 等待用户准备
            input("\n准备好后按回车开始采集...")
            
            # 采集数据
            measurements = self.collect_data(target_dist)
            if measurements is None:
                print("❌ 采集失败，跳过此点")
                continue
            
            # 评估数据
            result = self.evaluate_data(measurements, target_dist)
            
            # 询问是否保存
            if self.get_user_decision(result):
                # 保存（不保存测量值列表，减小文件大小）
                save_result = {
                    'true_distance': result['true_distance'],
                    'mean': result['mean'],
                    'std': result['std'],
                    'mae': result['mae'],
                    'error_percent': result['error_percent'],
                    'min': result['min'],
                    'max': result['max'],
                    'range': result['range'],
                    'samples': result['samples']
                }
                self.results.append(save_result)
                self.save_results()
                print("✅ 数据已保存")
            else:
                print("⏭️  数据未保存，将重新测量此点")
                # 重新测试这个点
                continue
            
            # 显示进度
            self.show_progress()
        
        # 打印最终汇总
        self.print_summary()
        self.cleanup()
    
    def print_summary(self):
        """打印最终汇总表格"""
        if not self.results:
            print("\n没有保存任何数据")
            return
        
        print("\n" + "="*70)
        print("📊 测距精度测试完成！")
        print("="*70)
        
        # 按距离排序
        sorted_results = sorted(self.results, key=lambda x: x['true_distance'])
        
        print(f"\n{'真值距离(cm)':<15} {'测量均值(cm)':<15} {'标准差(cm)':<12} {'MAE(cm)':<12}")
        print("-"*70)
        
        mae_list = []
        std_list = []
        
        for r in sorted_results:
            print(f"{r['true_distance']:<15} {r['mean']:<15.2f} {r['std']:<12.2f} {r['mae']:<12.2f}")
            mae_list.append(r['mae'])
            std_list.append(r['std'])
        
        print("-"*70)
        print(f"最大MAE: {max(mae_list):.2f} cm")
        print(f"平均MAE: {np.mean(mae_list):.2f} cm")
        print(f"最大标准差: {max(std_list):.2f} cm")
        print("="*70)
    
    def cleanup(self):
        GPIO.cleanup()

def main():
    tester = DistanceAccuracyTest()
    try:
        tester.run_test()
    finally:
        tester.cleanup()
        print("\n✅ GPIO已清理")

if __name__ == "__main__":
    main()
