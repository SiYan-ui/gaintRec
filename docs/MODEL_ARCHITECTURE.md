# Gait Recognition Model Architecture

本文档详细介绍了本项目中基于轮廓（Silhouette-based）的步态识别模型结构。该模型深受 GaitSet 启发，采用集合（Set）视角的处理方式，对帧的顺序具有一定的鲁棒性，并结合了时序差分特征。

## 1. 模型概览 (Overview)

整个模型由以下几个核心模块组成：
1.  **TemporalDifferencer**: 对输入视频帧序列进行多尺度时序差分，提取运动特征。
2.  **GaitSetBackbone**: 对每一帧（及其差分特征）独立进行卷积特征提取。
3.  **SetPooling**: 将时序维度的特征进行聚合，采用统计池化和金字塔池化，将变长的序列压缩为固定长度的嵌入向量。
4.  **Classifier**: 全连接层分类器，用于输出步态识别的类别概率。

---

## 2. 组件详解 (Components)

### 2.1 TemporalDifferencer (时序差分器)

该模块旨在通过计算帧与帧之间的差值来捕获运动信息。

*   **输入**: $ (B, T, C, H, W) $
    *   $B$: Batch size
    *   $T$: 时间序列长度 (帧数)
    *   $C$: 输入通道数 (通常为1，即二值轮廓图)
    *   $H, W$: 图像高宽
*   **逻辑**:
    *   对于给定的步长 `stride` 和最大差分距离 `diff_size`，计算当前帧 $X_t$ 与之前帧 $X_{t-k}$ 的差值。
    *   公式: $ D_k = X_t - X_{t-k} $ (通过 padding 0 处理边界)
    *   将原始输入和所有差分结果在通道维度拼接。
*   **输出**: $ (B, T, C_{out}, H, W) $
    *   通道数变大，包含了原始外观信息和运动差分信息。

### 2.2 GaitSetBackbone (骨干网络)

这是一个基于 2D CNN 的特征提取器，它**独立地**处理每一帧。

*   **输入**: $ (B, T, C_{in}, H, W) $ (来自 TemporalDifferencer 的输出)
*   **处理流程**:
    1.  **View**: 将 $B$ 和 $T$ 维度合并，视为 $ (B \times T, C_{in}, H, W) $ 进行批处理。
    2.  **CNN Layers**: 堆叠多个 `ConvBNRelu` 块和 `MaxPool2d` 层。
        *   `ConvBNRelu`: Conv2d (3x3) -> BatchNorm2d -> ReLU
    3.  **Global Average Pooling**: 使用 `AdaptiveAvgPool2d(1)` 将空间维度 $(H, W)$ 压缩为 $1 \times 1$。
    4.  **View**: 恢复时序维度，形状变为 $ (B, T, D) $。
*   **输出**: $ (B, T, D) $，其中 $D$ 是最后一层卷积的通道数。

### 2.3 SetPooling (集合池化)

这是 GaitSet 的核心思想所在，将帧特征序列聚合为一个全局特征向量。它包含三种池化策略：

*   **输入**: $ (B, T, D) $
*   **池化策略**:
    1.  **Global Mean Pooling**: 在 $T$ 维度取平均 $\rightarrow (B, D)$。
    2.  **Global Max Pooling**: 在 $T$ 维度取最大值 $\rightarrow (B, D)$。
    3.  **Temporal Pyramid Pooling (TPP)**:
        *   将 $T$ 维度划分为不同数量的 bins (例如 1, 2, 4)。
        *   对每个 bin 内的帧特征取平均。
        *   例如 `bins=4`，将序列分为4段，每段取平均，得到 4 个 $D$ 维向量。
        *   最后将所有 bins 的结果展平并拼接。
*   **输出**: $ (B, D_{total}) $
    *   $D_{total} = D \times (1_{mean} + 1_{max} + \sum bins)$

### 2.4 Classifier (分类头)

*   **Dropout**: 防止过拟合。
*   **Linear**: 将聚合后的高维特征映射到类别空间。
*   **输出**: $ (B, \text{num\_classes}) $

---

## 3. 数据流示意 (Data Flow)

假设配置如下：
*   Input: `(B, 30, 1, 64, 64)` (30帧, 64x64大小)
*   Backbone output channels: `128`
*   Pyramid bins: `(1, 2, 4)`

| 阶段 | 张量形状 (Shape) | 说明 |
| :--- | :--- | :--- |
| **Input** | `(B, 30, 1, 64, 64)` | 原始轮廓序列 |
| **TemporalDifferencer** | `(B, 30, C', 64, 64)` | 增加通道数以包含差分特征 |
| **Reshape (Backbone)** | `(B*30, C', 64, 64)` | 合并 Batch 和 Time 维度 |
| **CNN Layers** | `(B*30, 128, H', W')` | 空间特征提取 |
| **GAP (Spatial)** | `(B*30, 128, 1, 1)` | 空间维度压缩 |
| **Reshape (Restore)** | `(B, 30, 128)` | 恢复时序结构 |
| **SetPooling** | `(B, 1152)` | 聚合时序特征 |
| | | $128 \times (1 + 1 + 1 + 2 + 4) = 128 \times 9 = 1152$ |
| **Classifier** | `(B, num_classes)` | 得到分类 Logits |

---

## 4. 代码对应关系

*   `src/model/gaitset.py`:
    *   `TemporalDifferencer`: 对应第 2.1 节。
    *   `GaitSetBackbone`: 对应第 2.2 节。
    *   `SetPooling`: 对应第 2.3 节。
    *   `GaitRecognitionModel`: 对应整体组装。
