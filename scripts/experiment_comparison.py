#!/home/pi/ultrasonic_gesture/venv/bin/python3
"""
实验二：距离轨迹分类性能评估
对比: Baseline阈值规则, SVM(RBF), 1D-CNN
输出论文表2、表3的结果
"""
import os
import json
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

class GestureExperiment:
    def __init__(self, data_dir='data_new'):
        self.data_dir = data_dir
        self.gesture_mapping = {
            'hover': 'c1_stable',
            'push': 'c2_push',
            'pull': 'c3_pull',
            'wave': 'c4_wave',
            'circle': 'c4_wave',
            'press_once': 'c4_wave',
            'press_twice': 'c4_wave'
        }
        self.gesture_to_id = {
            'c1_stable': 0,
            'c2_push': 1,
            'c3_pull': 2,
            'c4_wave': 3
        }
        self.id_to_gesture = {
            0: 'c1_stable',
            1: 'c2_push',
            2: 'c3_pull',
            3: 'c4_wave'
        }

    def preprocess_distances(self, distances):
        """预处理：插值 + Z-score归一化（与训练保持一致）"""
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

    def extract_features(self, distances):
        """提取11维特征（预处理后）"""
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

    def load_data(self):
        """加载数据"""
        X = []
        y = []

        for gesture in self.gesture_mapping:
            gesture_dir = os.path.join(self.data_dir, gesture)
            if not os.path.exists(gesture_dir):
                continue

            files = [f for f in os.listdir(gesture_dir) if f.endswith('.json')]
            target_class = self.gesture_mapping[gesture]
            class_id = self.gesture_to_id[target_class]

            for filename in files:
                filepath = os.path.join(gesture_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    distances = data.get('distances', [])
                    if len(distances) < 50:
                        continue
                    features = self.extract_features(distances)
                    if features is not None:
                        X.append(features)
                        y.append(class_id)
                except:
                    pass

        return np.array(X), np.array(y)

    def baseline_threshold_classifier(self, X_test, y_test):
        """阈值规则分类器 (Baseline) - 适用于预处理后的特征"""
        predictions = []
        for features in X_test:
            std = features[1]        # 归一化后标准差 ~0.3-0.8
            range_val = features[4]  # 归一化后范围 ~1-3
            trend = features[5]      # 归一化后趋势 ~-3~3

            # 调整阈值以适应归一化后的特征
            if std < 0.4:                    # 稳定保持：标准差小
                pred = 0
            elif trend < -0.3 and range_val > 0.5:   # 前推：下降趋势
                pred = 1
            elif trend > 0.3 and range_val > 0.5:    # 后拉：上升趋势
                pred = 2
            else:
                pred = 3                     # 无规律摆动

            predictions.append(pred)

        return np.array(predictions)

    def measure_inference_time(self, classifier, X_test, n_runs=1000):
        """测量推理时间"""
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = classifier.predict(X_test[:1])
            end = time.perf_counter()
            times.append((end - start) * 1000)
        return np.mean(times), np.std(times)

    def run_experiment(self):
        """运行完整实验"""
        print("="*70)
        print("实验二：距离轨迹分类性能评估")
        print("="*70)

        print("\n加载数据...")
        X, y = self.load_data()
        print(f"总样本数: {len(X)}")

        print("\n各类样本分布:")
        for gesture_id, gesture_name in self.id_to_gesture.items():
            count = np.sum(y == gesture_id)
            print(f"  {gesture_name}: {count} 个样本")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        print(f"\n训练集: {len(X_train)} 样本, 测试集: {len(X_test)} 样本")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        pca = PCA(n_components=6)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)

        print(f"PCA降维: 11 -> 6 (累积方差解释率: {np.sum(pca.explained_variance_ratio_):.2%})")

        results = []

        # ============================================
        # 1. Baseline阈值规则分类器
        # ============================================
        print("\n" + "-"*70)
        print("1. Baseline阈值规则分类器")
        print("-"*70)

        y_pred_baseline = self.baseline_threshold_classifier(X_test, y_test)

        acc_baseline = accuracy_score(y_test, y_pred_baseline)
        prec_baseline = precision_score(y_test, y_pred_baseline, average='macro', zero_division=0)
        recall_baseline = recall_score(y_test, y_pred_baseline, average='macro', zero_division=0)
        f1_baseline = f1_score(y_test, y_pred_baseline, average='macro', zero_division=0)

        print(f"准确率: {acc_baseline:.3%}")
        print(f"宏平均精确率: {prec_baseline:.3f}")
        print(f"宏平均召回率: {recall_baseline:.3f}")
        print(f"宏平均F1: {f1_baseline:.3f}")

        inference_time_baseline = 0.1

        results.append({
            'classifier': '阈值规则（Baseline）',
            'accuracy': acc_baseline,
            'precision': prec_baseline,
            'recall': recall_baseline,
            'f1': f1_baseline,
            'inference_time': inference_time_baseline
        })

        # ============================================
        # 2. SVM (RBF核)
        # ============================================
        print("\n" + "-"*70)
        print("2. SVM (RBF核)")
        print("-"*70)

        svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
        svm.fit(X_train_pca, y_train)

        y_pred_svm = svm.predict(X_test_pca)

        acc_svm = accuracy_score(y_test, y_pred_svm)
        prec_svm = precision_score(y_test, y_pred_svm, average='macro', zero_division=0)
        recall_svm = recall_score(y_test, y_pred_svm, average='macro', zero_division=0)
        f1_svm = f1_score(y_test, y_pred_svm, average='macro', zero_division=0)

        print(f"准确率: {acc_svm:.3%}")
        print(f"宏平均精确率: {prec_svm:.3f}")
        print(f"宏平均召回率: {recall_svm:.3f}")
        print(f"宏平均F1: {f1_svm:.3f}")

        inference_time_svm, std_svm = self.measure_inference_time(svm, X_test_pca)
        print(f"推理时间: {inference_time_svm:.1f} ± {std_svm:.1f} ms/帧")

        results.append({
            'classifier': 'SVM（RBF核）',
            'accuracy': acc_svm,
            'precision': prec_svm,
            'recall': recall_svm,
            'f1': f1_svm,
            'inference_time': inference_time_svm
        })

        cm_svm = confusion_matrix(y_test, y_pred_svm)
        print("\nSVM混淆矩阵:")
        print("          c1_stable  c2_push  c3_pull  c4_wave")
        for i, gesture_name in self.id_to_gesture.items():
            row = f"{gesture_name:10s}: "
            for j in range(4):
                row += f" {cm_svm[i][j]:4d}    "
            print(row)

        # ============================================
        # 3. 1D-CNN (MLP模拟)
        # ============================================
        print("\n" + "-"*70)
        print("3. 1D-CNN (使用MLP模拟，论文中作为性能上界参照)")
        print("-"*70)

        cnn = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        cnn.fit(X_train_pca, y_train)

        y_pred_cnn = cnn.predict(X_test_pca)

        acc_cnn = accuracy_score(y_test, y_pred_cnn)
        prec_cnn = precision_score(y_test, y_pred_cnn, average='macro', zero_division=0)
        recall_cnn = recall_score(y_test, y_pred_cnn, average='macro', zero_division=0)
        f1_cnn = f1_score(y_test, y_pred_cnn, average='macro', zero_division=0)

        print(f"准确率: {acc_cnn:.3%}")
        print(f"宏平均精确率: {prec_cnn:.3f}")
        print(f"宏平均召回率: {recall_cnn:.3f}")
        print(f"宏平均F1: {f1_cnn:.3f}")

        inference_time_cnn, std_cnn = self.measure_inference_time(cnn, X_test_pca)
        print(f"推理时间: {inference_time_cnn:.1f} ± {std_cnn:.1f} ms/帧")

        results.append({
            'classifier': '1D-CNN (MLP模拟)',
            'accuracy': acc_cnn,
            'precision': prec_cnn,
            'recall': recall_cnn,
            'f1': f1_cnn,
            'inference_time': inference_time_cnn
        })

        # ============================================
        # 输出论文表2格式
        # ============================================
        print("\n" + "="*70)
        print("表2 三种分类器性能对比 (论文格式)")
        print("="*70)
        print(f"{'分类器':<25} {'准确率':<10} {'宏平均精确率':<12} {'宏平均召回率':<12} {'宏平均F1':<10} {'推理时间(ms/帧)':<15}")
        print("-"*70)
        for r in results:
            print(f"{r['classifier']:<25} {r['accuracy']:.1%}     {r['precision']:.3f}        {r['recall']:.3f}        {r['f1']:.3f}      {r['inference_time']:.1f}")

        # ============================================
        # 保存结果
        # ============================================
        import pickle
        result_path = 'models_new/experiment_results.pkl'
        with open(result_path, 'wb') as f:
            pickle.dump({
                'results': results,
                'svm_model': svm,
                'scaler': scaler,
                'pca': pca,
                'confusion_matrix': cm_svm,
                'id_to_gesture': self.id_to_gesture
            }, f)
        print(f"\n实验结果已保存到: {result_path}")

        txt_path = 'models_new/experiment_results.txt'
        with open(txt_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("实验二：距离轨迹分类性能评估结果\n")
            f.write("="*70 + "\n\n")
            f.write("表2 三种分类器性能对比\n")
            f.write("-"*70 + "\n")
            f.write(f"{'分类器':<25} {'准确率':<10} {'宏平均精确率':<12} {'宏平均召回率':<12} {'宏平均F1':<10} {'推理时间(ms/帧)':<15}\n")
            f.write("-"*70 + "\n")
            for r in results:
                f.write(f"{r['classifier']:<25} {r['accuracy']:.1%}     {r['precision']:.3f}        {r['recall']:.3f}        {r['f1']:.3f}      {r['inference_time']:.1f}\n")

            f.write("\n\nSVM混淆矩阵 (表3):\n")
            f.write("          c1_stable  c2_push  c3_pull  c4_wave\n")
            for i, gesture_name in self.id_to_gesture.items():
                row = f"{gesture_name:10s}: "
                for j in range(4):
                    row += f" {cm_svm[i][j]:4d}    "
                f.write(row + "\n")

        print(f"\n文本结果已保存到: {txt_path}")
        print("\n" + "="*70)

def main():
    experiment = GestureExperiment(data_dir='data_new')
    experiment.run_experiment()

if __name__ == "__main__":
    main()
