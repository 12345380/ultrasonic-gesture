#!/home/pi/ultrasonic_gesture/venv/bin/python3
"""
消融实验 - 7种手势版本
使用原始7种手势，验证领域特征的有效性
"""
import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

class AblationStudy7Gestures:
    def __init__(self, data_dir='data_new'):
        self.data_dir = data_dir
        # 使用7种原始手势
        self.gestures = ['hover', 'push', 'pull', 'wave', 'circle', 'press_once', 'press_twice']
        self.gesture_to_id = {g: i for i, g in enumerate(self.gestures)}
        self.id_to_gesture = {i: g for i, g in enumerate(self.gestures)}
        
    def extract_all_features(self, distances):
        """提取全部11维特征"""
        dist_array = np.array(distances)
        L = len(dist_array)
        if L < 10:
            return None
        
        features = []
        # F1-F8: 通用特征
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
            features.append(np.sum((dist_array[2:] - 2 * dist_array[1:-1] + dist_array[:-2]) ** 2))
        else:
            features.append(0)
        
        return np.array(features)
    
    def extract_features_without(self, distances, remove_indices):
        """提取特征，移除指定索引"""
        all_features = self.extract_all_features(distances)
        if all_features is None:
            return None
        return np.delete(all_features, remove_indices)
    
    def load_data(self, feature_mode='all'):
        """加载数据"""
        X = []
        y = []
        
        remove_config = {
            'all': [],
            'without_jitter': [8],
            'without_consistency': [9],
            'without_smoothness': [10],
            'without_domain': [8, 9, 10],
            'general_only': [8, 9, 10]
        }
        
        remove_indices = remove_config.get(feature_mode, [])
        
        for gesture in self.gestures:
            gesture_dir = os.path.join(self.data_dir, gesture)
            if not os.path.exists(gesture_dir):
                continue
            
            files = [f for f in os.listdir(gesture_dir) if f.endswith('.json')]
            class_id = self.gesture_to_id[gesture]
            
            for filename in files:
                filepath = os.path.join(gesture_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    distances = data.get('distances', [])
                    if len(distances) < 50:
                        continue
                    
                    if remove_indices:
                        features = self.extract_features_without(distances, remove_indices)
                    else:
                        features = self.extract_all_features(distances)
                    
                    if features is not None:
                        X.append(features)
                        y.append(class_id)
                except:
                    pass
        
        return np.array(X), np.array(y)
    
    def train_and_evaluate(self, X, y):
        """训练SVM并评估"""
        if len(X) == 0:
            return None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        n_components = min(6, X.shape[1])
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        
        svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        svm.fit(X_train_pca, y_train)
        
        y_pred = svm.predict(X_test_pca)
        
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        return {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'n_samples': len(X),
            'n_features': X.shape[1],
            'n_train': len(X_train),
            'n_test': len(X_test)
        }
    
    def run(self):
        """运行消融实验"""
        print("="*70)
        print("消融实验: 7种手势版本")
        print("="*70)
        print("\n实验目的:")
        print("  在7种手势分类任务上验证领域特征的有效性")
        print("  由于类别更多，特征贡献会更明显")
        print("-"*70)
        
        experiments = [
            ('all', '全部特征 (11维)'),
            ('without_jitter', '移除F9 (抖动指数)'),
            ('without_consistency', '移除F10 (意图一致性)'),
            ('without_smoothness', '移除F11 (动作平滑度)'),
            ('without_domain', '移除全部领域特征'),
            ('general_only', '仅通用特征 (F1-F8)')
        ]
        
        results = {}
        
        for mode, description in experiments:
            print(f"\n📊 运行: {description}")
            
            X, y = self.load_data(mode)
            if len(X) == 0:
                print(f"   ⚠️  无数据，跳过")
                continue
            
            result = self.train_and_evaluate(X, y)
            if result is None:
                print(f"   ⚠️  训练失败，跳过")
                continue
            
            results[mode] = result
            
            print(f"   样本数: {result['n_samples']}")
            print(f"   特征维度: {result['n_features']}")
            print(f"   测试集: {result['n_test']} 样本")
            print(f"   准确率: {result['accuracy']:.2%}")
            print(f"   宏平均F1: {result['f1']:.4f}")
        
        self.print_summary(results)
        self.save_results(results)
    
    def print_summary(self, results):
        """打印汇总表格"""
        print("\n" + "="*70)
        print("表: 消融实验结果 (7种手势)")
        print("="*70)
        print(f"{'特征配置':<30} {'特征数':<8} {'准确率':<10} {'宏平均F1':<12} {'F1下降':<10}")
        print("-"*70)
        
        baseline_f1 = results.get('all', {}).get('f1', 0)
        
        for mode, description in [
            ('all', '全部特征 (11维)'),
            ('without_jitter', '移除F9 (抖动指数)'),
            ('without_consistency', '移除F10 (意图一致性)'),
            ('without_smoothness', '移除F11 (动作平滑度)'),
            ('without_domain', '移除全部领域特征'),
            ('general_only', '仅通用特征 (F1-F8)')
        ]:
            r = results.get(mode)
            if r is None:
                continue
            
            f1 = r['f1']
            acc = r['accuracy']
            
            if mode == 'all':
                f1_drop = 0
            else:
                f1_drop = (baseline_f1 - f1) * 100
            
            drop_str = f"{f1_drop:+.1f}%"
            if mode != 'all' and f1_drop > 1:
                drop_str = f"⬇️ {f1_drop:.1f}%"
            
            print(f"{description:<30} {r['n_features']:<8} {acc:.2%}     {f1:.4f}       {drop_str}")
        
        print("="*70)
        
        # 结论
        print("\n📝 结论:")
        all_result = results.get('all')
        without_domain = results.get('without_domain')
        
        if all_result and without_domain:
            drop = (all_result['f1'] - without_domain['f1']) * 100
            print(f"  ✅ 全部特征宏平均F1 = {all_result['f1']:.4f}")
            print(f"  ⬇️ 移除全部领域特征后F1 = {without_domain['f1']:.4f}")
            print(f"  📉 下降: {drop:.1f}个百分点")
        
        if all_result:
            drops = {}
            for mode in ['without_jitter', 'without_consistency', 'without_smoothness']:
                r = results.get(mode)
                if r:
                    drops[mode] = (all_result['f1'] - r['f1']) * 100
            
            if drops:
                print("\n  🔑 各特征贡献排序:")
                sorted_drops = sorted(drops.items(), key=lambda x: x[1], reverse=True)
                name_map = {
                    'without_jitter': '抖动指数 (F9)',
                    'without_consistency': '意图一致性 (F10)',
                    'without_smoothness': '动作平滑度 (F11)'
                }
                for i, (mode, drop) in enumerate(sorted_drops, 1):
                    if drop > 0.1:
                        print(f"    {i}. {name_map[mode]}: 贡献{drop:.1f}个百分点")
                    else:
                        print(f"    {i}. {name_map[mode]}: 贡献微小")
        print("="*70)
    
    def save_results(self, results):
        """保存结果"""
        result_dir = 'experiment_results'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        
        txt_path = os.path.join(result_dir, 'ablation_results_7gestures.txt')
        with open(txt_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("消融实验结果 (7种手势)\n")
            f.write("="*70 + "\n\n")
            
            baseline_f1 = results.get('all', {}).get('f1', 0)
            
            f.write(f"{'特征配置':<30} {'特征数':<8} {'准确率':<10} {'宏平均F1':<12} {'F1下降':<10}\n")
            f.write("-"*70 + "\n")
            
            for mode, description in [
                ('all', '全部特征 (11维)'),
                ('without_jitter', '移除F9 (抖动指数)'),
                ('without_consistency', '移除F10 (意图一致性)'),
                ('without_smoothness', '移除F11 (动作平滑度)'),
                ('without_domain', '移除全部领域特征'),
                ('general_only', '仅通用特征 (F1-F8)')
            ]:
                r = results.get(mode)
                if r is None:
                    continue
                
                f1 = r['f1']
                acc = r['accuracy']
                
                if mode == 'all':
                    f1_drop = 0
                else:
                    f1_drop = (baseline_f1 - f1) * 100
                
                f.write(f"{description:<30} {r['n_features']:<8} {acc:.2%}     {f1:.4f}       {f1_drop:+.1f}%\n")
            
            f.write("="*70 + "\n")
        
        print(f"\n✅ 结果已保存到: {txt_path}")

def main():
    study = AblationStudy7Gestures(data_dir='data_new')
    study.run()

if __name__ == "__main__":
    main()
