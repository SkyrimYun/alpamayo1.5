# Alpamayo 1.5 主模型的 Forward 函数位置

## 回答

在这个仓库中，主模型 **Alpamayo1_5** 没有显式定义传统的 `forward()` 函数。

### 主模型类位置
- **文件**: `src/alpamayo1_5/models/alpamayo1_5.py`
- **类名**: `Alpamayo1_5` (第82行)

### 继承结构
```
Alpamayo1_5
  └─> ReasoningVLA (在 src/alpamayo1_5/models/base_model.py 第289行)
      └─> PreTrainedModel (HuggingFace transformers)
      └─> TrajectoryFusionMixin (在 base_model.py 第129行)
```

### Forward 函数的位置

#### 1. 主模型推理方法（代替传统 forward）
主模型使用以下方法进行推理，而不是传统的 `forward()` 函数：

- **`sample_trajectories_from_data_with_vlm_rollout`** (第214行)
  - 主要的推理方法
  - 生成轨迹预测，结合 VLM 推理和扩散采样

- **`sample_trajectories_from_data_with_vlm_rollout_cfg_nav`** (第404行)
  - 带导航引导的推理方法
  - 使用 Classifier-Free Guidance (CFG)

- **`generate_text`** (在 base_model.py 第451行)
  - 纯文本生成方法
  - 用于视觉问答 (VQA)

#### 2. 子模块的 Forward 函数
仓库中有以下子模块定义了 `forward()` 函数：

**文件**: `src/alpamayo1_5/models/action_in_proj.py`
- `RMSNorm.forward()` (第32行) - 归一化层
- `MLPEncoder.forward()` (第68行) - MLP编码器
- `FourierEncoderV2.forward()` (第91行) - 傅里叶特征编码
- `PerWaypointActionInProjV2.forward()` (第148行) - 动作投影模块

### 为什么没有显式的 forward 函数？

Alpamayo1_5 是一个复杂的混合模型，包含：
1. **VLM (视觉-语言模型)**: 使用 Qwen3VL 作为骨干网络
2. **Expert 网络**: 用于轨迹生成的专家模型
3. **扩散模型**: 用于动作空间采样

推理流程是通过专门的方法实现的，而不是简单的 forward 传播：
1. VLM 自回归生成推理文本 (chain-of-causation)
2. Expert 网络使用扩散过程生成轨迹
3. 使用 KV 缓存进行高效推理

### 使用示例

```python
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

model = Alpamayo1_5.from_pretrained("nvidia/Alpamayo-1.5-10B")

# 使用主要推理方法（不是 forward）
pred_xyz, pred_rot = model.sample_trajectories_from_data_with_vlm_rollout(
    data=data,
    num_traj_samples=16
)

# 或使用文本生成
text_outputs = model.generate_text(data=data)
```

## 总结

主模型的 "forward" 功能分布在：
1. **主推理入口**: `Alpamayo1_5.sample_trajectories_from_data_with_vlm_rollout()`
   - 位置: `src/alpamayo1_5/models/alpamayo1_5.py:214`
2. **VLM forward**: 由内部的 `self.vlm` (Qwen3VL) 处理
3. **Expert forward**: 由内部的 `self.expert` 处理 (第342行)
4. **子模块 forward**: 在 `src/alpamayo1_5/models/action_in_proj.py` 中定义
