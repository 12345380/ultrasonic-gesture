#!/home/pi/ultrasonic_gesture/venv/bin/python3
"""
识别4种手势 (论文版本) - 包含预处理
c1: 稳定保持
c2: 前推
c3: 后拉
c4: 无规律摆动
"""
import RPi.GPIO as GPIO
import time
import numpy as np
import pickle
import os
from collections import deque

class FourGestureRecognizer:
    def __init__(self, model_path='models_new/four_gesture_model.pkl'):
        self.model = None
        self.scaler = None
        self.pca = None
        self.id_to_gesture = None
        self.gesture_to_id = None
        self.gesture_descriptions = None

        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)

                self.model = model_data['svm']
                self.scaler = model_data['scaler']
                self.pca = model_data['pca']
                self.id_to_gesture = model_data['id_to_gesture']
                self.gesture_to_id = model_data['gesture_to_id']
                self.gesture_descriptions = model_data.get('gesture_descriptions', {})

                print("="*60)
                print("4种手势识别器 (论文版本)")
                print("="*60)
                print("✓ 模型加载成功")
                print("✓ 支持 4 种手势:")
                for gesture_id, gesture_name in sorted(self.id_to_gesture.items()):
                    desc = self.gesture_descriptions.get(gesture_name, '')
                    print(f"  {gesture_id}: {gesture_name:12s} - {desc}")

            except Exception as e:
                print(f"❌ 模型加载失败: {e}")
                self.model = None
        else:
            print("❌ 未找到模型文件")
            print("请先运行: python scripts/train_four_gestures.py")
            self.model = None

        self.TRIG = 23
        self.ECHO = 24

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.TRIG, GPIO.OUT)
        GPIO.setup(self.ECHO, GPIO.IN)

        self.data_buffer = deque(maxlen=150)

        print("\n原始手势 -> 论文4类:")
        print("  hover      -> c1_stable (稳定保持)")
        print("  push       -> c2_push   (前推)")
        print("  pull       -> c3_pull   (后拉)")
        print("  wave       -> c4_wave   (无规律摆动)")
        print("  circle     -> c4_wave   (无规律摆动)")
        print("  press_once -> c4_wave   (无规律摆动)")
        print("  press_twice-> c4_wave   (无规律摆动)")
        print("="*60)

    def get_distance(self):
        """获取单次距离测量"""
        GPIO.output(self.TRIG, False)
        time.sleep(0.005)

        GPIO.output(self.TRIG, True)
        time.sleep(0.00001)
        GPIO.output(self.TRIG, False)

        pulse_start = time.time()
        timeout = pulse_start + 0.04

        while GPIO.input(self.ECHO) == 0 and pulse_start < timeout:
            pulse_start = time.time()

        if pulse_start >= timeout:
            return None

        pulse_end = time.time()
        while GPIO.input(self.ECHO) == 1 and pulse_end < timeout:
            pulse_end = time.time()

        if pulse_end >= timeout:
            return None

        pulse_duration = pulse_end - pulse_start
        distance = pulse_duration * 17150

        if 2 <= distance <= 400:
            return distance
        return None

    def preprocess_distances(self, distances):
        """预处理：插值 + Z-score归一化（与训练保持一致）"""
        dist_array = np.array(distances, dtype=float)
        L = len(dist_array)
        if L < 10:
            return None

        # 1. 线性插值补齐缺失帧
        if np.any(np.isnan(dist_array)):
            x = np.arange(len(dist_array))
            mask = ~np.isnan(dist_array)
            if np.sum(mask) >= 2:
                dist_array = np.interp(x, x[mask], dist_array[mask])
            else:
                return None

        # 2. Z-score归一化（零均值，单位方差）
        mean_val = np.mean(dist_array)
        std_val = np.std(dist_array)
        if std_val > 1e-6:
            dist_array = (dist_array - mean_val) / std_val

        return dist_array

    def extract_features(self, distances):
        """提取11维特征（预处理后）"""
        # 先做预处理
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
            diff = np.diff(dist_array)
            features.append(np.mean(diff))
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
            jerk = np.sum((dist_array[2:] - 2 * dist_array[1:-1] + dist_array[:-2]) ** 2)
            features.append(jerk)
        else:
            features.append(0)

        return np.array(features)

    def recognize_gesture(self, distances):
        """识别手势"""
        if self.model is None:
            return "无模型", 0.0, {}

        try:
            features = self.extract_features(distances)
            if features is None:
                return "数据不足", 0.0, {}

            features_scaled = self.scaler.transform([features])
            features_pca = self.pca.transform(features_scaled)

            prediction = self.model.predict(features_pca)[0]
            gesture_name = self.id_to_gesture[prediction]

            probabilities = self.model.predict_proba(features_pca)[0]
            confidence = probabilities[prediction]

            gesture_probs = {}
            for i, prob in enumerate(probabilities):
                if i in self.id_to_gesture:
                    gesture_probs[self.id_to_gesture[i]] = prob

            return gesture_name, confidence, gesture_probs

        except Exception as e:
            return f"识别错误: {e}", 0.0, {}

    def single_recognition(self):
        """单次手势识别"""
        print("\n" + "="*60)
        print("单次手势识别模式")
        print("="*60)
        print("说明:")
        print("1. 按回车开始采集（3秒）")
        print("2. 在传感器前做手势")
        print("3. 显示识别结果")
        print("4. 按 Ctrl+C 退出程序")
        print("-"*60)

        print("\n手势说明:")
        for gesture_id, gesture_name in sorted(self.id_to_gesture.items()):
            desc = self.gesture_descriptions.get(gesture_name, '')
            print(f"  {gesture_name}: {desc}")
        print("-"*60)

        try:
            while True:
                input("\n按回车开始采集手势（3秒）... ")

                print("采集数据中...")

                distances = []
                start_time = time.time()

                while time.time() - start_time < 3.0:
                    dist = self.get_distance()
                    if dist:
                        distances.append(dist)
                        print(f"距离: {dist:5.1f} cm | 样本: {len(distances):3d}", end='\r')
                    time.sleep(0.02)

                print()

                if len(distances) < 50:
                    print("⚠️  数据不足（<50个样本），请重试")
                    continue

                gesture, confidence, probs = self.recognize_gesture(distances)

                print("\n" + "="*60)
                print(f"识别结果: {gesture}")
                print(f"置信度: {confidence:.2%}")

                if gesture in self.gesture_descriptions:
                    print(f"说明: {self.gesture_descriptions[gesture]}")

                print("-"*60)

                dist_array = np.array(distances)
                print(f"数据统计:")
                print(f"  样本数: {len(distances)}")
                print(f"  平均距离: {np.mean(dist_array):.1f} cm")
                print(f"  距离范围: {np.min(dist_array):.1f} - {np.max(dist_array):.1f} cm")
                print(f"  标准差: {np.std(dist_array):.1f} cm")

                print("\n所有手势概率:")
                sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
                for gesture_name, prob in sorted_probs:
                    if prob > 0.01:
                        bar_length = int(prob * 40)
                        bar = "█" * bar_length + " " * (40 - bar_length)
                        desc = self.gesture_descriptions.get(gesture_name, '')[:15]
                        print(f"  {gesture_name:12s} {bar} {prob:.1%}  {desc}")

                print("="*60)

        except KeyboardInterrupt:
            print("\n退出程序")

    def demo_mode(self):
        """演示模式：连续识别"""
        print("\n" + "="*60)
        print("演示模式 - 连续识别")
        print("="*60)
        print("按 Ctrl+C 退出")
        print("-"*60)

        try:
            while True:
                distances = []
                start_time = time.time()

                while time.time() - start_time < 1.5:
                    dist = self.get_distance()
                    if dist:
                        distances.append(dist)
                    time.sleep(0.02)

                if len(distances) < 25:
                    continue

                gesture, confidence, _ = self.recognize_gesture(distances)

                if confidence > 0.5:
                    print(f"\r识别: {gesture:12s} (置信度: {confidence:.1%})", end='')

                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n退出演示模式")

    def run(self):
        """运行识别器"""
        if self.model is None:
            print("无法运行识别器，请先训练模型")
            print("运行: python scripts/train_four_gestures.py")
            return

        print("\n选择模式:")
        print("  1. 单次识别模式 (采集3秒后识别)")
        print("  2. 演示模式 (连续识别)")
        print("  3. 退出")

        while True:
            try:
                choice = input("\n请输入选择 (1-3): ").strip()

                if choice == '1':
                    self.single_recognition()
                elif choice == '2':
                    self.demo_mode()
                elif choice == '3':
                    print("退出程序")
                    break
                else:
                    print("无效选择，请重新输入")
            except KeyboardInterrupt:
                print("\n退出程序")
                break

        GPIO.cleanup()

def main():
    recognizer = FourGestureRecognizer()
    recognizer.run()

if __name__ == "__main__":
    main()
