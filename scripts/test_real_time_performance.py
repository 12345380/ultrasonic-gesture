#!/home/pi/ultrasonic_gesture/venv/bin/python3
"""
实验三：系统端到端实时性能测试 (RQ3) - 修正版
每个手势给3-5秒时间完成
"""
import RPi.GPIO as GPIO
import time
import numpy as np
import pickle
import os
import json
from collections import deque
import random

TRIG = 23
ECHO = 24

class RealTimePerformanceTest:
    def __init__(self, model_path='models_new/four_gesture_model.pkl'):
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(TRIG, GPIO.OUT)
        GPIO.setup(ECHO, GPIO.IN)

        self.model = None
        self.scaler = None
        self.pca = None
        self.id_to_gesture = None
        self.gesture_descriptions = {
            0: 'c1_stable (稳定保持)',
            1: 'c2_push (前推)',
            2: 'c3_pull (后拉)',
            3: 'c4_wave (无规律摆动)'
        }

        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            self.model = model_data['svm']
            self.scaler = model_data['scaler']
            self.pca = model_data['pca']
            self.id_to_gesture = model_data['id_to_gesture']
            print("✅ 模型加载成功")

        self.n_tests = 500
        self.sample_rate = 50
        self.sample_interval = 1.0 / self.sample_rate
        self.gesture_duration = 3.0

        self.data_buffer = deque(maxlen=150)
        self.latencies = []
        self.latency_details = {
            'sampling': [],
            'filtering': [],
            'feature_extraction': [],
            'inference': [],
            'game_logic': []
        }
        self.gesture_history = []

        self.result_dir = 'experiment_results'
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

    def get_distance(self):
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

    def preprocess_distances(self, distances):
        """预处理：插值 + Z-score归一化"""
        dist_array = np.array(distances, dtype=float)
        L = len(dist_array)
        if L < 10:
            return None

        if np.any(np.isnan(dist_array)):
            x = np.arange(len(dist_array))
            mask = ~np.isnan(dist_array)
            if np.sum(mask) >= 2:
                dist_array = np.interp(x, x[mask], dist_array[mask])
            else:
                return None

        mean_val = np.mean(dist_array)
        std_val = np.std(dist_array)
        if std_val > 1e-6:
            dist_array = (dist_array - mean_val) / std_val

        return dist_array

    def median_filter(self, data, window_size=5):
        if len(data) < window_size:
            return data[-1] if data else 0
        window = list(data)[-window_size:]
        return np.median(window)

    def extract_features(self, distances):
        dist_array = self.preprocess_distances(distances)
        if dist_array is None:
            return None

        L = len(dist_array)
        features = []
        features.append(np.mean(dist_array))
        features.append(np.std(dist_array))
        features.append(np.max(dist_array))
        features.append(np.min(dist_array))
        features.append(np.ptp(dist_array))

        if L > 1:
            features.append(np.mean(np.diff(dist_array)))
        else:
            features.append(0)

        if L > 1:
            zcr = np.sum(np.diff(np.sign(dist_array - np.mean(dist_array))) != 0) / (L - 1)
            features.append(zcr)
        else:
            features.append(0)

        if np.std(dist_array) > 0:
            features.append(np.mean(((dist_array - np.mean(dist_array)) / np.std(dist_array)) ** 3))
        else:
            features.append(0)

        if L > 1:
            features.append(np.median(np.abs(np.diff(dist_array))))
        else:
            features.append(0)

        if L > 1:
            diff = np.diff(dist_array)
            up = np.sum(diff > 0.3)
            down = np.sum(diff < -0.3)
            stable = np.sum(np.abs(diff) <= 0.3)
            features.append(max(up, down, stable) / L)
        else:
            features.append(0)

        if L > 2:
            features.append(np.sum((dist_array[2:] - 2 * dist_array[1:-1] + dist_array[:-2]) ** 2))
        else:
            features.append(0)

        return np.array(features)

    def simulate_game_logic(self, gesture_id):
        time.sleep(0.001)
        gesture_map = {0: '蓄力', 1: '跳跃', 2: '下蹲', 3: '无操作'}
        return gesture_map.get(gesture_id, '未知')

    def get_gesture_instruction(self, gesture_id):
        instructions = {
            0: '🖐️  稳定保持: 手静止在20-30cm处，保持不动',
            1: '👉  前推: 手从40cm匀速推向10cm',
            2: '👈  后拉: 手从10cm匀速拉回40cm',
            3: '🔄  无规律摆动: 手快速前后无序摆动'
        }
        return instructions.get(gesture_id, '未知手势')

    def measure_end_to_end_latency(self):
        print("\n" + "="*70)
        print("测量端到端延迟...")
        print("="*70)
        print(f"总测试次数: {self.n_tests}")
        print(f"每个手势持续: {self.gesture_duration} 秒")
        print("\n系统会提示你做一个手势，请保持该手势3秒钟")
        print("之后系统会自动切换到下一个手势")
        print("-"*70)

        gesture_sequence = []
        for i in range(self.n_tests // 4 + 1):
            gesture_sequence.extend([0, 1, 2, 3])
        gesture_sequence = gesture_sequence[:self.n_tests]
        random.shuffle(gesture_sequence)

        total_collected = 0
        current_gesture_idx = 0

        try:
            while total_collected < self.n_tests:
                target_gesture = gesture_sequence[current_gesture_idx]
                gesture_name = self.gesture_descriptions[target_gesture]
                instruction = self.get_gesture_instruction(target_gesture)

                print(f"\n{'='*60}")
                print(f"📢 请做手势: {gesture_name}")
                print(f"   {instruction}")
                print(f"   请持续做这个手势 {self.gesture_duration} 秒...")
                print(f"   采集倒计时: ", end='')

                for t in range(int(self.gesture_duration), 0, -1):
                    print(f"{t}... ", end='', flush=True)
                    time.sleep(1)
                print("开始采集! 🔴")

                start_time = time.time()
                gesture_samples = 0

                while time.time() - start_time < self.gesture_duration:
                    sampling_start = time.perf_counter()
                    time.sleep(self.sample_interval)
                    dist = self.get_distance()
                    sampling_end = time.perf_counter()
                    sampling_time = (sampling_end - sampling_start) * 1000

                    if dist is not None:
                        self.data_buffer.append(dist)

                    if len(self.data_buffer) < 20:
                        continue

                    filter_start = time.perf_counter()
                    filtered = self.median_filter(self.data_buffer, window_size=5)
                    filter_end = time.perf_counter()
                    filter_time = (filter_end - filter_start) * 1000

                    feature_start = time.perf_counter()
                    features = self.extract_features(list(self.data_buffer))
                    feature_end = time.perf_counter()
                    feature_time = (feature_end - feature_start) * 1000

                    if features is None:
                        continue

                    inference_start = time.perf_counter()
                    features_scaled = self.scaler.transform([features])
                    features_pca = self.pca.transform(features_scaled)
                    prediction = self.model.predict(features_pca)[0]
                    probabilities = self.model.predict_proba(features_pca)[0]
                    confidence = probabilities[prediction]
                    inference_end = time.perf_counter()
                    inference_time = (inference_end - inference_start) * 1000

                    game_start = time.perf_counter()
                    action = self.simulate_game_logic(prediction)
                    game_end = time.perf_counter()
                    game_time = (game_end - game_start) * 1000

                    total_latency = sampling_time + filter_time + feature_time + inference_time + game_time

                    self.latencies.append(total_latency)
                    self.latency_details['sampling'].append(sampling_time)
                    self.latency_details['filtering'].append(filter_time)
                    self.latency_details['feature_extraction'].append(feature_time)
                    self.latency_details['inference'].append(inference_time)
                    self.latency_details['game_logic'].append(game_time)
                    self.gesture_history.append({
                        'target': target_gesture,
                        'predicted': prediction,
                        'confidence': confidence
                    })

                    gesture_samples += 1
                    total_collected += 1

                    pred_name = self.id_to_gesture.get(prediction, 'unknown')
                    print(f"\r   样本: {total_collected}/{self.n_tests} | 距离: {dist:5.1f}cm | 识别: {pred_name:15s} | 延迟: {total_latency:.1f}ms", end='')

                print(f"\n   ✅ 完成 {gesture_name} 采集，共 {gesture_samples} 个样本")
                current_gesture_idx += 1

                if current_gesture_idx < len(gesture_sequence):
                    print("\n⏳ 准备下一个手势...")
                    time.sleep(1)

        except KeyboardInterrupt:
            print("\n⚠️  测试被中断")

        print(f"\n\n✅ 采集完成！共 {total_collected} 个样本")

    def calculate_statistics(self):
        if not self.latencies:
            print("没有有效的延迟数据")
            return None

        latencies = np.array(self.latencies)
        stats = {
            'mean': np.mean(latencies),
            'std': np.std(latencies),
            'min': np.min(latencies),
            'max': np.max(latencies),
            'p50': np.percentile(latencies, 50),
            'p95': np.percentile(latencies, 95),
            'p99': np.percentile(latencies, 99),
            'n_samples': len(latencies)
        }
        return stats

    def calculate_detail_stats(self):
        detail_stats = {}
        for name, values in self.latency_details.items():
            if values:
                arr = np.array(values)
                detail_stats[name] = {
                    'mean': np.mean(arr),
                    'std': np.std(arr),
                    'min': np.min(arr),
                    'max': np.max(arr),
                    'p95': np.percentile(arr, 95)
                }
        return detail_stats

    def print_results(self, stats, detail_stats):
        print("\n" + "="*70)
        print("实验三：系统端到端实时性能测试结果")
        print("="*70)

        print("\n端到端延迟统计 (ms):")
        print("-"*50)
        print(f"  均值:     {stats['mean']:.1f} ms")
        print(f"  标准差:   {stats['std']:.1f} ms")
        print(f"  最小值:   {stats['min']:.1f} ms")
        print(f"  最大值:   {stats['max']:.1f} ms")
        print(f"  P50:      {stats['p50']:.1f} ms")
        print(f"  P95:      {stats['p95']:.1f} ms")
        print(f"  P99:      {stats['p99']:.1f} ms")
        print(f"  样本数:   {stats['n_samples']}")

        print("\n端到端延迟构成分解:")
        print("-"*50)
        total_mean = stats['mean']
        for name, d in detail_stats.items():
            percentage = (d['mean'] / total_mean) * 100
            print(f"  {name:20s}: {d['mean']:.1f} ms ({percentage:.1f}%)")

        print("\n" + "="*70)
        print("表3 端到端实时性能测试结果 (论文格式)")
        print("="*70)
        print(f"{'统计量':<20} {'延迟(ms)':<15}")
        print("-"*50)
        print(f"{'均值':<20} {stats['mean']:.1f}")
        print(f"{'标准差':<20} {stats['std']:.1f}")
        print(f"{'P95 (第95百分位数)':<20} {stats['p95']:.1f}")
        print(f"{'P99':<20} {stats['p99']:.1f}")
        print(f"{'最大值':<20} {stats['max']:.1f}")
        print("="*70)

        self.save_results(stats, detail_stats)

    def save_results(self, stats, detail_stats):
        json_path = os.path.join(self.result_dir, 'real_time_performance_results.json')
        result_data = {
            'end_to_end': {
                'mean': stats['mean'],
                'std': stats['std'],
                'min': stats['min'],
                'max': stats['max'],
                'p50': stats['p50'],
                'p95': stats['p95'],
                'p99': stats['p99'],
                'n_samples': stats['n_samples']
            },
            'detail_breakdown': {
                name: {
                    'mean': d['mean'],
                    'std': d['std'],
                    'min': d['min'],
                    'max': d['max'],
                    'p95': d['p95']
                } for name, d in detail_stats.items()
            }
        }

        with open(json_path, 'w') as f:
            json.dump(result_data, f, indent=2)
        print(f"\n✅ 结果已保存到: {json_path}")

        txt_path = os.path.join(self.result_dir, 'real_time_performance_table.txt')
        with open(txt_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("实验三：系统端到端实时性能测试结果\n")
            f.write("="*70 + "\n\n")
            f.write("表3 端到端实时性能测试结果\n")
            f.write("-"*50 + "\n")
            f.write(f"{'统计量':<20} {'延迟(ms)':<15}\n")
            f.write("-"*50 + "\n")
            f.write(f"{'均值':<20} {stats['mean']:.1f}\n")
            f.write(f"{'标准差':<20} {stats['std']:.1f}\n")
            f.write(f"{'P95':<20} {stats['p95']:.1f}\n")
            f.write(f"{'P99':<20} {stats['p99']:.1f}\n")
            f.write(f"{'最大值':<20} {stats['max']:.1f}\n")
            f.write("\n延迟构成分解:\n")
            f.write("-"*50 + "\n")
            total_mean = stats['mean']
            for name, d in detail_stats.items():
                percentage = (d['mean'] / total_mean) * 100
                f.write(f"{name:20s}: {d['mean']:.1f} ms ({percentage:.1f}%)\n")

        print(f"✅ 文本结果已保存到: {txt_path}")

    def run_test(self):
        print("="*70)
        print("实验三：系统端到端实时性能测试 (RQ3)")
        print("="*70)
        print("\n实验说明:")
        print("1. 系统会提示你做4种手势之一")
        print("2. 每个手势持续3秒，请保持动作")
        print("3. 系统自动采集数据并测量延迟")
        print("4. 共采集500个样本")
        print("5. 延迟目标: < 50ms")
        print("-"*70)

        if self.model is None:
            print("\n❌ 模型未加载，请先训练模型")
            return

        input("\n准备好后按回车开始测试...")

        print("\n预热中...")
        for _ in range(50):
            self.get_distance()
        print("✅ 预热完成")

        self.measure_end_to_end_latency()

        stats = self.calculate_statistics()
        detail_stats = self.calculate_detail_stats()

        if stats:
            self.print_results(stats, detail_stats)

        if stats and stats['p99'] < 50:
            print("\n✅ 系统满足实时交互需求 (P99 < 50ms)")
        else:
            print(f"\n⚠️  系统延迟略高 (P99 = {stats['p99']:.1f}ms)")

    def cleanup(self):
        GPIO.cleanup()

def main():
    tester = RealTimePerformanceTest()
    try:
        tester.run_test()
    finally:
        tester.cleanup()
        print("\n✅ GPIO已清理")

if __name__ == "__main__":
    main()
