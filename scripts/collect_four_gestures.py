#!/home/pi/ultrasonic_gesture/venv/bin/python3
"""
采集4种手势数据 (论文版本)
数据保存到 data_new/ 目录
c1: 稳定保持 (hover)
c2: 前推 (push)
c3: 后拉 (pull)
c4: 无规律摆动 (wave)
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

class FourGestureCollector:
    def __init__(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(TRIG, GPIO.OUT)
        GPIO.setup(ECHO, GPIO.IN)
        
        # 4种手势定义 (论文版本)
        self.gestures = {
            '1': 'hover',       # c1_stable: 稳定保持
            '2': 'push',        # c2_push: 前推
            '3': 'pull',        # c3_pull: 后拉
            '4': 'wave'         # c4_wave: 无规律摆动
        }
        
        # 手势映射到论文类别
        self.gesture_to_paper = {
            'hover': 'c1_stable',
            'push': 'c2_push',
            'pull': 'c3_pull',
            'wave': 'c4_wave'
        }
        
        # 采集参数
        self.sample_rate = 50  # Hz
        self.duration = 3.0    # 秒
        self.samples_per_gesture = int(self.sample_rate * self.duration)
        
        # 数据保存到 data_new/ 目录 (与训练脚本一致)
        self.data_dir = 'data_new'
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        # 为每个手势创建子目录
        for gesture in self.gestures.values():
            gesture_dir = os.path.join(self.data_dir, gesture)
            if not os.path.exists(gesture_dir):
                os.makedirs(gesture_dir)
    
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
    
    def show_gesture_instruction(self, gesture_name):
        """显示手势动作说明"""
        instructions = {
            'hover': """
╔══════════════════════════════════════════════════════════════╗
║  🖐️  c1_stable: 稳定保持 (hover)                            ║
╠══════════════════════════════════════════════════════════════╣
║  动作: 手静止悬停在传感器前                                  ║
║  距离: 保持20-30cm                                          ║
║  稳定性: 尽量保持不动                                        ║
║  时间: 持续3秒                                              ║
║  要点: 手不要晃动，呼吸要平稳                                ║
║  判断标准: 距离变化 < 2cm                                    ║
╚══════════════════════════════════════════════════════════════╝
            """,
            
            'push': """
╔══════════════════════════════════════════════════════════════╗
║  🖐️  c2_push: 前推 (push)                                   ║
╠══════════════════════════════════════════════════════════════╣
║  动作: 手从远到近匀速靠近传感器                              ║
║  距离: 40cm → 10cm                                          ║
║  速度: 中等（3秒完成）                                       ║
║  要点: 直线移动，不要晃动                                    ║
║  判断标准: 距离单调递减，下降 > 5cm                          ║
╚══════════════════════════════════════════════════════════════╝
            """,
            
            'pull': """
╔══════════════════════════════════════════════════════════════╗
║  🖐️  c3_pull: 后拉 (pull)                                   ║
╠══════════════════════════════════════════════════════════════╣
║  动作: 手从近到远匀速远离传感器                              ║
║  距离: 10cm → 40cm                                          ║
║  速度: 中等（3秒完成）                                       ║
║  要点: 直线移动，不要晃动                                    ║
║  判断标准: 距离单调递增，上升 > 5cm                          ║
╚══════════════════════════════════════════════════════════════╝
            """,
            
            'wave': """
╔══════════════════════════════════════════════════════════════╗
║  🖐️  c4_wave: 无规律摆动 (wave)                             ║
╠══════════════════════════════════════════════════════════════╣
║  动作: 手无节奏地前后快速摆动                                ║
║  距离: 保持20-30cm范围内摆动                                 ║
║  幅度: 前后各10-15cm                                         ║
║  频率: 2-3次/秒（要快！）                                    ║
║  要点: 手腕发力，无规律、不重复                              ║
║  判断标准: 高频大幅无序波动                                  ║
╚══════════════════════════════════════════════════════════════╝
            """
        }
        
        print(instructions.get(gesture_name, f"未知手势: {gesture_name}"))
    
    def collect_gesture(self, gesture_name):
        """采集单个手势数据"""
        paper_class = self.gesture_to_paper[gesture_name]
        
        print(f"\n{'='*60}")
        print(f"采集手势: {gesture_name} -> {paper_class}")
        print(f"{'='*60}")
        
        # 显示详细说明
        self.show_gesture_instruction(gesture_name)
        
        # 显示采集参数
        print(f"\n采集参数:")
        print(f"  采样率: {self.sample_rate} Hz")
        print(f"  采集时长: {self.duration} 秒")
        print(f"  预计样本数: {self.samples_per_gesture}")
        print(f"  保存目录: {self.data_dir}/{gesture_name}/")
        
        # 显示已有样本数
        gesture_dir = os.path.join(self.data_dir, gesture_name)
        existing_files = [f for f in os.listdir(gesture_dir) if f.endswith('.json')]
        print(f"  已有样本: {len(existing_files)} 个")
        
        # 等待开始信号
        input("\n按回车键开始采集...")
        
        print("\n采集开始！请做手势...")
        print("按 Ctrl+C 可中断采集\n")
        
        distances = []
        timestamps = []
        
        start_time = time.time()
        sample_count = 0
        
        # 显示实时采集进度
        print("进度: [", end='')
        progress_width = 50
        
        try:
            while sample_count < self.samples_per_gesture:
                dist = self.get_distance()
                if dist is not None:
                    distances.append(dist)
                    current_time = time.time() - start_time
                    timestamps.append(current_time)
                    sample_count += 1
                    
                    # 更新进度条
                    progress = int(sample_count / self.samples_per_gesture * progress_width)
                    bar = "█" * progress + "░" * (progress_width - progress)
                    print(f"\r进度: [{bar}] {sample_count}/{self.samples_per_gesture}", end='')
                    
                    # 显示实时距离
                    print(f" | 距离: {dist:5.1f} cm", end='')
                
                # 控制采样率
                time.sleep(1.0 / self.sample_rate)
                
        except KeyboardInterrupt:
            print("\n\n⚠️  采集被中断")
            if len(distances) < 30:
                print("数据太少，放弃保存")
                return None
        
        print(f"\n\n采集完成！共采集 {len(distances)} 个数据点")
        
        if len(distances) < 30:
            print("⚠️  数据点太少，建议重新采集")
            return None
        
        # 创建数据记录
        data = {
            'gesture': gesture_name,
            'paper_class': paper_class,
            'timestamp': datetime.now().isoformat(),
            'sample_rate': self.sample_rate,
            'duration': self.duration,
            'distances': distances,
            'timestamps': timestamps
        }
        
        return data
    
    def validate_data(self, data):
        """验证采集的数据质量"""
        distances = np.array(data['distances'])
        gesture = data['gesture']
        
        if len(distances) < 50:
            return False, "数据点不足（<50）"
        
        stats = {
            'mean': np.mean(distances),
            'std': np.std(distances),
            'min': np.min(distances),
            'max': np.max(distances),
            'range': np.max(distances) - np.min(distances)
        }
        
        print(f"\n📊 数据统计:")
        print(f"  平均值: {stats['mean']:.2f} cm")
        print(f"  标准差: {stats['std']:.2f} cm")
        print(f"  最小值: {stats['min']:.2f} cm")
        print(f"  最大值: {stats['max']:.2f} cm")
        print(f"  变化范围: {stats['range']:.2f} cm")
        
        # 计算趋势
        quarter = len(distances) // 4
        if quarter > 0:
            trend = np.mean(distances[-quarter:]) - np.mean(distances[:quarter])
            print(f"  趋势（后-前）: {trend:.2f} cm")
        
        # 手势特定验证
        issues = []
        
        if gesture == 'hover':
            if stats['std'] > 2.0:
                issues.append(f"稳定保持应该有很小的变化（标准差={stats['std']:.2f}cm > 2cm）")
            if stats['range'] > 5.0:
                issues.append(f"稳定保持的变化范围应该很小（范围={stats['range']:.2f}cm > 5cm）")
                
        elif gesture == 'push':
            quarter = len(distances) // 4
            if quarter > 0:
                trend = np.mean(distances[-quarter:]) - np.mean(distances[:quarter])
                if trend > -3:
                    issues.append(f"推动作应该有明显的下降趋势（趋势={trend:.2f}cm）")
            if stats['range'] < 8:
                issues.append(f"推动作的变化范围应该较大（范围={stats['range']:.2f}cm < 8cm）")
                
        elif gesture == 'pull':
            quarter = len(distances) // 4
            if quarter > 0:
                trend = np.mean(distances[-quarter:]) - np.mean(distances[:quarter])
                if trend < 3:
                    issues.append(f"拉动作应该有明显的上升趋势（趋势={trend:.2f}cm）")
            if stats['range'] < 8:
                issues.append(f"拉动作的变化范围应该较大（范围={stats['range']:.2f}cm < 8cm）")
                
        elif gesture == 'wave':
            if stats['std'] < 3:
                issues.append(f"无规律摆动应该有较大的变化（标准差={stats['std']:.2f}cm < 3cm）")
            if stats['range'] < 8:
                issues.append(f"无规律摆动的变化范围应该较大（范围={stats['range']:.2f}cm < 8cm）")
        
        if issues:
            print(f"\n⚠️  潜在问题:")
            for issue in issues:
                print(f"  • {issue}")
            return False, "数据质量可能有问题"
        else:
            return True, "数据质量良好 ✓"
    
    def save_data(self, data):
        """保存采集的数据到 data_new/"""
        gesture = data['gesture']
        paper_class = data['paper_class']
        
        # 保存到 data_new/gesture/ 目录
        gesture_dir = os.path.join(self.data_dir, gesture)
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{gesture}_{paper_class}_{timestamp}.json"
        
        filepath = os.path.join(gesture_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n✅ 数据已保存到: {filepath}")
        return filepath
    
    def show_collected_stats(self):
        """显示已采集的数据统计"""
        print(f"\n{'='*60}")
        print(f"已采集数据统计 (目录: {self.data_dir}/)")
        print(f"{'='*60}")
        
        total_files = 0
        for gesture in self.gestures.values():
            gesture_dir = os.path.join(self.data_dir, gesture)
            if os.path.exists(gesture_dir):
                files = [f for f in os.listdir(gesture_dir) if f.endswith('.json')]
                count = len(files)
                total_files += count
                paper_class = self.gesture_to_paper[gesture]
                print(f"  {gesture:12s} -> {paper_class:12s}: {count:3d} 个样本")
        
        print(f"{'-'*60}")
        print(f"  总计: {total_files} 个样本")
        print(f"\n训练命令: python scripts/train_four_gestures.py")
    
    def run_collection(self):
        """运行数据采集主程序"""
        print("="*60)
        print("4种手势数据采集系统 (论文版本)")
        print(f"数据保存目录: {self.data_dir}/")
        print("="*60)
        print("\n采集的手势:")
        print("  1. hover  -> c1_stable (稳定保持)")
        print("  2. push   -> c2_push   (前推)")
        print("  3. pull   -> c3_pull   (后拉)")
        print("  4. wave   -> c4_wave   (无规律摆动)")
        
        print("\n命令:")
        print("  q: 退出程序")
        print("  s: 显示已采集的数据统计")
        print("  c: 继续采集")
        print("  l: 重新显示手势列表")
        
        while True:
            print(f"\n{'='*60}")
            command = input("请输入手势编号 (1-4) 或命令 (q/s/c/l): ").strip().lower()
            
            if command == 'q':
                print("退出程序...")
                break
            
            elif command == 's':
                self.show_collected_stats()
            
            elif command == 'c':
                continue
            
            elif command == 'l':
                print("\n可采集的手势:")
                for key, gesture in self.gestures.items():
                    paper_class = self.gesture_to_paper[gesture]
                    print(f"  {key}. {gesture:12s} -> {paper_class}")
                continue
            
            elif command in self.gestures:
                gesture_name = self.gestures[command]
                
                # 采集数据
                data = self.collect_gesture(gesture_name)
                
                if data is None:
                    print("采集失败，请重试")
                    continue
                
                # 验证数据质量
                is_valid, message = self.validate_data(data)
                
                if is_valid:
                    print(f"\n✅ 数据验证: {message}")
                else:
                    print(f"\n⚠️  数据验证: {message}")
                    confirm = input("是否仍然保存？(y/n): ").strip().lower()
                    if confirm != 'y':
                        print("数据未保存")
                        continue
                
                # 保存数据
                save_option = input("是否保存数据？(y/n): ").strip().lower()
                if save_option == 'y':
                    self.save_data(data)
                    self.show_collected_stats()
                else:
                    print("数据未保存")
            
            else:
                print("无效输入，请重试")
    
    def cleanup(self):
        """清理GPIO资源"""
        GPIO.cleanup()

def main():
    collector = FourGestureCollector()
    
    try:
        collector.run_collection()
    finally:
        collector.cleanup()
        print("\nGPIO已清理")

if __name__ == "__main__":
    main()
