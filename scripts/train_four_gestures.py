#!/home/pi/ultrasonic_gesture/venv/bin/python3
"""
训练4种手势识别模型（论文版本）
c1: 稳定保持
c2: 前推
c3: 后拉
c4: 无规律摆动
"""
import os
import json
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class FourGestureTrainer:
    def __init__(self, data_dir='data_new'):
        self.data_dir = data_dir
        # 4种手势映射（论文中的4类）
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
        self.gesture_descriptions = {
            'c1_stable': '稳定保持: 距离变化<2cm，持续0.5s以上',
            'c2_push': '前推: 距离单调递减，下降>5cm',
            'c3_pull': '后拉: 距离单调递增，上升>5cm',
            'c4_wave': '无规律摆动: 高频大幅无序波动'
        }

    def preprocess_distances(self, distances):
        """预处理：插值 + Z-score归一化"""
        dist_array = np.array(distances, dtype=float)
        L = len(dist_array)
        if L < 10:
            return None

        # 1. 线性插值补齐缺失帧（如果有None/NaN）
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

        # F1: 均值
        features.append(np.mean(dist_array))
        # F2: 标准差
        features.append(np.std(dist_array))
        # F3: 最大值
        features.append(np.max(dist_array))
        # F4: 最小值
        features.append(np.min(dist_array))
        # F5: 变化幅度
        features.append(np.ptp(dist_array))

        # F6: 一阶差分均值
        if L > 1:
            diff = np.diff(dist_array)
            features.append(np.mean(diff))
        else:
            features.append(0)

        # F7: 过零率
        if L > 1:
            zcr = np.sum(np.diff(np.sign(dist_array - np.mean(dist_array))) != 0) / (L - 1)
            features.append(zcr)
        else:
            features.append(0)

        # F8: 偏度
        if np.std(dist_array) > 0:
            features.append(np.mean(((dist_array - np.mean(dist_array)) / np.std(dist_array)) ** 3))
        else:
            features.append(0)

        # F9: 抖动指数
        if L > 1:
            features.append(np.median(np.abs(np.diff(dist_array))))
        else:
            features.append(0)

        # F10: 意图一致性
        if L > 1:
            diff = np.diff(dist_array)
            up = np.sum(diff > 0.3)
            down = np.sum(diff < -0.3)
            stable = np.sum(np.abs(diff) <= 0.3)
            features.append(max(up, down, stable) / L)
        else:
            features.append(0)

        # F11: 动作平滑度
        if L > 2:
            jerk = np.sum((dist_array[2:] - 2 * dist_array[1:-1] + dist_array[:-2]) ** 2)
            features.append(jerk)
        else:
            features.append(0)

        return np.array(features)

    def load_data(self):
        """加载数据"""
        print("="*60)
        print("加载手势数据 (4类)")
        print("="*60)

        X = []
        y = []
        gesture_counts = {}

        for gesture in self.gesture_mapping:
            gesture_dir = os.path.join(self.data_dir, gesture)
            if not os.path.exists(gesture_dir):
                print(f"⚠️  {gesture} 目录不存在")
                continue

            files = [f for f in os.listdir(gesture_dir) if f.endswith('.json')]
            if not files:
                print(f"⚠️  {gesture} 目录下没有数据文件")
                continue

            target_class = self.gesture_mapping[gesture]
            class_id = self.gesture_to_id[target_class]

            print(f"处理 {gesture} -> {target_class} ({len(files)} 个文件): ", end='')
            count = 0

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
                        count += 1
                except Exception as e:
                    print(f"E", end='')

            print(f" → {count} 个样本")
            if target_class in gesture_counts:
                gesture_counts[target_class] += count
            else:
                gesture_counts[target_class] = count

        if not X:
            print("❌ 没有加载到有效数据")
            return None, None, None

        print("\n" + "="*60)
        print("数据统计")
        print("="*60)
        for gesture_id, gesture_name in self.id_to_gesture.items():
            count = gesture_counts.get(gesture_name, 0)
            desc = self.gesture_descriptions.get(gesture_name, '')
            print(f"{gesture_name:12s} (ID={gesture_id}): {count:3d} 个样本 - {desc}")

        total = len(X)
        print(f"\n总样本数: {total}")
        print(f"特征维度: {len(X[0])}")

        return np.array(X), np.array(y)

    def train_model(self, X, y):
        """训练模型 (SVM + PCA)"""
        print("\n" + "="*60)
        print("训练SVM分类器")
        print("="*60)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"训练集: {len(X_train)} 个样本")
        print(f"测试集: {len(X_test)} 个样本")

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        pca = PCA(n_components=6)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)

        print(f"PCA降维: {X.shape[1]} -> 6 (累积方差解释率: {np.sum(pca.explained_variance_ratio_):.2%})")

        svm = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=42
        )
        svm.fit(X_train_pca, y_train)

        train_acc = svm.score(X_train_pca, y_train)
        test_acc = svm.score(X_test_pca, y_test)

        print(f"\n模型性能:")
        print(f"  训练集准确率: {train_acc:.4f} ({train_acc*100:.1f}%)")
        print(f"  测试集准确率: {test_acc:.4f} ({test_acc*100:.1f}%)")

        y_pred = svm.predict(X_test_pca)
        print("\n每个手势的分类性能:")
        for gesture_id, gesture_name in self.id_to_gesture.items():
            mask = y_test == gesture_id
            if np.sum(mask) > 0:
                class_acc = np.mean(y_pred[mask] == y_test[mask])
                print(f"  {gesture_name:12s}: {class_acc:.3f} ({np.sum(mask)} 个样本)")

        print("\n混淆矩阵:")
        cm = confusion_matrix(y_test, y_pred)
        print("          c1_stable  c2_push  c3_pull  c4_wave")
        for i, gesture_name in self.id_to_gesture.items():
            row = f"{gesture_name:10s}: "
            for j in range(4):
                row += f" {cm[i][j]:4d}    "
            print(row)

        return {
            'svm': svm,
            'scaler': scaler,
            'pca': pca,
            'test_acc': test_acc,
            'X_train': X_train_pca,
            'y_train': y_train
        }

    def save_model(self, model_data, output_dir='models_new'):
        """保存模型"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        model_pkg = {
            'svm': model_data['svm'],
            'scaler': model_data['scaler'],
            'pca': model_data['pca'],
            'id_to_gesture': self.id_to_gesture,
            'gesture_to_id': self.gesture_to_id,
            'gesture_descriptions': self.gesture_descriptions,
            'feature_count': 11,
            'pca_components': 6,
            'model_type': 'SVM_RBF_PCA'
        }

        model_path = os.path.join(output_dir, 'four_gesture_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model_pkg, f)

        print(f"\n模型已保存到: {model_path}")

        info_path = os.path.join(output_dir, 'model_info_four.txt')
        with open(info_path, 'w') as f:
            f.write("4种手势识别模型 (论文版本)\n")
            f.write("="*50 + "\n")
            f.write(f"创建时间: {np.datetime64('now')}\n")
            f.write(f"手势数量: 4\n")
            f.write(f"模型类型: SVM (RBF核) + PCA\n")
            f.write(f"特征维度: 11 -> 6 (PCA)\n")
            f.write(f"测试集准确率: {model_data['test_acc']:.2%}\n")
            f.write("\n支持的手势:\n")
            for gesture_id, gesture_name in sorted(self.id_to_gesture.items()):
                desc = self.gesture_descriptions.get(gesture_name, '')
                f.write(f"  {gesture_id}: {gesture_name} - {desc}\n")

        return model_path

    def run(self):
        """运行训练流程"""
        print("="*60)
        print("4种手势识别模型训练 (论文版本)")
        print("手势: c1_stable, c2_push, c3_pull, c4_wave")
        print("="*60)

        X, y = self.load_data()
        if X is None:
            return

        model_data = self.train_model(X, y)
        model_path = self.save_model(model_data)

        print("\n" + "="*60)
        print("训练完成!")
        print("="*60)
        print(f"模型文件: {model_path}")
        print("运行识别: python scripts/recognize_four_gestures.py")
        print("="*60)

def main():
    trainer = FourGestureTrainer(data_dir='data_new')
    trainer.run()

if __name__ == "__main__":
    main()
