# Ultrasonic Gesture Recognition System

基于树莓派和HC-SR04超声波传感器的低成本体感交互系统，用于儿童专注力训练。

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi-green.svg)](https://www.raspberrypi.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📄 论文信息

本仓库是论文 **《基于超声波距离轨迹识别的儿童体感交互系统设计与性能评估》** 的完整代码实现。

| 指标 | 结果 |
|------|------|
| 硬件成本 | **< 100元** |
| 手势识别 | **4种轨迹模式**（稳定保持、前推、后拉、无规律摆动） |
| 分类准确率 | **98.1%** |
| 宏平均F1 | **0.973** |
| 响应延迟 | **32.9ms**（P99 = 34.9ms） |
| 测距精度 | 最大MAE **0.33cm**，平均MAE **0.14cm** |
| 抗干扰 | 所有偏移 **< 0.1cm**，异常率 **0%** |

## 🏗️ 系统架构

基于树莓派和HC-SR04超声波传感器的低成本体感交互系统，用于儿童专注力训练。

## 论文信息

本仓库是论文 **《基于超声波距离轨迹识别的儿童体感交互系统设计与性能评估》** 的完整代码实现。

- **硬件成本**: < 100元
- **手势识别**: 4种轨迹模式（稳定保持、前推、后拉、无规律摆动）
- **分类准确率**: 98.1%
- **响应延迟**: 32.9ms (P99 = 34.9ms)

## 系统架构
┌─────────────────────────────────────────────────────────────────┐
│ 应 用 层 │
│ ┌─────────────────────────────┐ │
│ │ 接物下落游戏 │ │
│ │ （距离→接盘位置映射） │ │
│ └─────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│ 推 理 层 │
│ ┌─────────────────────────────┐ │
│ │ SVM分类器 (RBF核) │ │
│ │ 6维PCA输入 │ │
│ │ 推理时间: 0.3ms │ │
│ └─────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│ 处 理 层 │
│ ┌─────────────────────────────┐ │
│ │ 11维特征提取 │ │
│ │ + 中值滤波(窗口5) │ │
│ │ + Z-score归一化 │ │
│ └─────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│ 感 知 层 │
│ ┌─────────────────────────────┐ │
│ │ HC-SR04传感器 │ │
│ │ 50Hz采样 │ │
│ │ 10-60cm有效范围 │ │
│ └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘

## 硬件连接

| HC-SR04 | Raspberry Pi |
|---------|--------------|
| VCC     | 5V (Pin 2/4) |
| GND     | GND (Pin 6)  |
| Trig    | GPIO 23 (Pin 16) |
| Echo    | GPIO 24 (Pin 18) |

## 快速开始
1. 安装依赖
pip install RPi.GPIO numpy scikit-learn matplotlib pygame
2. 数据采集
如需采集新数据：python scripts/collect_four_gestures.py
3. 训练模型
python scripts/train_four_gestures.py
4. 运行手势识别
python scripts/recognize_four_gestures.py
5. 运行游戏
python scripts/game_catching.py
6. 项目结构
ultrasonic_gesture/
├── scripts/                    # 核心代码
│   ├── train_four_gestures.py      # 训练模型
│   ├── recognize_four_gestures.py  # 手势识别
│   ├── game_catching.py            # 游戏
│   ├── experiment_comparison.py    # 实验二：分类对比
│   ├── test_distance_accuracy.py   # 实验一：测距精度
│   ├── test_real_time_performance.py # 实验三：实时性能
│   ├── test_anti_interference.py   # 实验四：抗干扰
│   ├── ablation_study.py           # 消融实验
│   └── plot_*.py                   # 绘图
├── data_new/                   # 手势数据集
├── models_new/                 # 训练好的模型
└── experiment_results/         # 实验结果
7. 🎮 游戏说明
游戏为 “接物下落” ，通过超声波距离控制底部接盘左右移动：
🟢 绿色球（好球）：接住得 +1 分
🔴 红色球（坏球）：接到扣 -1 分
⏱ 游戏时长：120秒
🎯 控制方式：手靠近传感器 → 接盘右移；手远离传感器 → 接盘左移
