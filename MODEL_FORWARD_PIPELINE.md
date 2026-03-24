# Alpamayo 1.5 主推理流程详解

## 方法：`sample_trajectories_from_data_with_vlm_rollout_cfg_nav`

本文档详细解析带有 Classifier-Free Guidance (CFG) 和导航条件的轨迹采样推理流程。

**文件位置**: `src/alpamayo1_5/models/alpamayo1_5.py:404`

---

## 一、输入数据 (Inputs)

### 1. 必需输入参数

```python
data: dict[str, Any]  # 包含以下关键字段：
```

**data 字典的关键字段：**

- **`ego_history_xyz`**: `torch.Tensor [B, n_traj_group, T_hist, 3]`
  - 自车历史轨迹的 XYZ 坐标
  - B: 批次大小
  - n_traj_group: 轨迹组数（推理时必须为 1）
  - T_hist: 历史时间步数

- **`ego_history_rot`**: `torch.Tensor [B, n_traj_group, T_hist, 4]`
  - 自车历史轨迹的旋转（四元数）

- **`tokenized_data`**: `dict` 包含：
  - **`input_ids`**: `torch.Tensor [B, L]` - 输入的 token IDs
  - **`attention_mask`**: `torch.Tensor [B, L]` - 注意力掩码
  - **`pixel_values`**: `torch.Tensor` - 图像像素值（多相机输入）
  - **`image_grid_thw`**: `torch.Tensor` - 图像网格的时间、高度、宽度信息

### 2. 采样控制参数

- **`num_traj_samples`**: `int = 6` - 每个输入生成的轨迹样本数
- **`num_traj_sets`**: `int = 1` - 轨迹集合数
- **`top_p`**: `float = 0.98` - nucleus 采样的累积概率阈值
- **`top_k`**: `int | None` - top-k 采样的 k 值
- **`temperature`**: `float = 0.6` - 采样温度，控制随机性
- **`diffusion_kwargs`**: `dict | None` - 传递给扩散模型的额外参数

### 3. 可选参数

- **`max_generation_length`**: 在 `kwargs` 中，控制 VLM 生成的最大 token 数
- **`return_extra`**: 在 `kwargs` 中，如果为 True，返回额外的文本信息

---

## 二、主推理步骤 (Main Inference Steps)

### 阶段 0: 数据预处理 (第 433-446 行)

**作用**: 准备输入数据并融合轨迹 tokens

```python
# 深拷贝数据，避免修改原始数据
data = copy.deepcopy(data)
n_samples_total = num_traj_samples * num_traj_sets  # 总样本数

# 提取历史轨迹
ego_history_xyz = data["ego_history_xyz"]  # [B, 1, T_hist, 3]
ego_history_rot = data["ego_history_rot"]  # [B, 1, T_hist, 4]

# 提取并融合 token
tokenized_data = data["tokenized_data"]
input_ids = tokenized_data.pop("input_ids")
# 将历史轨迹编码为 tokens 并融合到 input_ids 中
input_ids = self.fuse_traj_tokens(input_ids, traj_data_vlm)
```

**关键操作**:
- 验证 `n_traj_group == 1`（推理时只支持单个轨迹组）
- `fuse_traj_tokens`: 将历史轨迹离散化为 token 并插入到输入序列中

---

### 阶段 1: VLM 自回归生成 (第 448-493 行)

**作用**: 使用 VLM (Vision-Language Model) 生成推理文本（Chain-of-Causation）

```python
# 配置生成参数
generation_config.top_p = top_p
generation_config.temperature = temperature
generation_config.do_sample = True
generation_config.num_return_sequences = num_traj_samples
generation_config.max_new_tokens = max_generation_length

# 自回归生成
vlm_outputs = self.vlm.generate(
    input_ids=input_ids,
    generation_config=generation_config,
    stopping_criteria=stopping_criteria,
    logits_processor=logits_processor,
    **tokenized_data,
)
```

**关键机制**:
1. **停止条件**: 在 `<traj_future_start>` token 后停止生成
2. **Logits 处理**: `ExpertLogitsProcessor` 屏蔽离散轨迹 tokens，确保 VLM 只生成文本
3. **输出**: 生成的 token 序列和 KV 缓存（`past_key_values`）

**VLM 输出**:
- `vlm_outputs.sequences`: `[B * num_traj_samples, L_gen]` - 生成的完整序列
- `vlm_outputs.past_key_values`: KV 缓存，用于后续 Expert 模型
- `vlm_outputs.rope_deltas`: RoPE 位置编码的偏移量

**后处理**:
- 清理生成的 logits 以节省内存
- 替换 EOS token 后的 padding
- 保存 KV 缓存作为 `prompt_cache`（guided）

---

### 阶段 2: 构建无引导 KV 缓存 (第 515-596 行)

**作用**: 为 Classifier-Free Guidance (CFG) 准备无导航信息的 KV 缓存

**步骤 2.1: 移除导航文本 (第 517-526 行)**
```python
# 从 input_ids 中移除 <route_start>...<route_end> 之间的导航信息
unguided_input_ids = []
for i in range(input_ids.shape[0]):
    unguided_input_ids.append(remove_nav_text(input_ids, self.tokenizer, i)[0])
# 重新填充为统一长度
unguided_input_ids = pad_sequence(unguided_input_ids, ...)
```

**步骤 2.2: 预填充无引导前缀 (第 528-537 行)**
```python
unguided_prefill_outputs = self.vlm(
    input_ids=unguided_input_ids,
    attention_mask=unguided_prefix_mask,
    image_grid_thw=tokenized_data.get("image_grid_thw"),
    pixel_values=tokenized_data.get("pixel_values"),
    use_cache=True,
)
```

**关键优化**:
- 视觉编码器只运行一次（基于原始 batch B）
- 不需要重复 `pixel_values`

**步骤 2.3: 重复 KV 缓存 (第 540-544 行)**
```python
unguided_prompt_cache = unguided_prefill_outputs.past_key_values
unguided_prompt_cache.batch_repeat_interleave(n_samples_total)
```

**步骤 2.4: 前向传播生成的 tokens (第 546-573 行)**
```python
# 使用已重复的 KV 缓存处理每个样本不同的生成 tokens
generated_tokens = vlm_outputs.sequences[:, input_ids.shape[1]:]
unguided_vlm_outputs = self.vlm(
    input_ids=generated_tokens,
    attention_mask=full_attention_mask,
    past_key_values=unguided_prompt_cache,
    cache_position=cache_position,
    use_cache=True,
)
```

**步骤 2.5: 构建无引导的位置和掩码 (第 575-596 行)**
```python
# 为 Expert 模型构建位置 IDs 和注意力掩码
unguided_position_ids, unguided_attention_mask = self._build_expert_pos_ids_and_attn_mask(
    offset=unguided_offset,
    rope_deltas=unguided_vlm_outputs.rope_deltas,
    kv_cache_seq_len=unguided_prompt_cache.get_seq_length(),
    n_diffusion_tokens=n_diffusion_tokens,
    b_star=b_star,
    device=device,
    prefix_mask=unguided_prefix_mask_repeated,
)
```

---

### 阶段 3: 定义去噪步骤函数 (第 602-636 行)

**作用**: 定义 Expert 模型的单步去噪函数，用于扩散采样

```python
def step_fn(
    x: torch.Tensor,          # [B*, T_f, C_action] 噪声动作
    t: torch.Tensor,          # [B*, 1, 1] 时间步
    position_ids: torch.Tensor,
    past_key_values: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    # 1. 将噪声动作投影为 token embeddings
    future_token_embeds = self.action_in_proj(x, t)  # [B*, T_f, hidden_size]

    # 2. Expert 模型前向传播
    expert_out = self.expert(
        inputs_embeds=future_token_embeds,
        position_ids=position_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        use_cache=True,
    )

    # 3. 投影回动作空间
    last_hidden = expert_out.last_hidden_state[:, -n_diffusion_tokens:]
    pred = self.action_out_proj(last_hidden)  # [B*, T_f, C_action]

    return pred  # 返回速度场（velocity field）
```

**关键组件**:
- **`action_in_proj`**: 将动作 + 时间步编码为 Expert 的输入 embeddings
  - 使用 Fourier 特征编码时间步
  - 使用 MLP 编码器处理动作特征

- **`expert`**: Transformer 解码器（与 VLM 相同架构）
  - 使用 VLM 的 KV 缓存进行交叉注意力
  - 只处理未来的 diffusion tokens

- **`action_out_proj`**: 将 hidden states 投影回动作空间

---

### 阶段 4: 扩散采样 (第 638-660 行)

**作用**: 使用 Flow Matching 在动作空间中采样轨迹

```python
sampled_action = self.diffusion.sample(
    batch_size=total_batch,
    step_fn=partial(
        step_fn,
        past_key_values=prompt_cache,          # 带导航引导
        attention_mask=attention_mask,
        position_ids=position_ids,
    ),
    unguided_step_fn=partial(
        step_fn,
        past_key_values=unguided_prompt_cache,  # 无导航引导
        attention_mask=unguided_attention_mask,
        position_ids=unguided_position_ids,
    ),
    device=device,
    return_all_steps=False,
    **diffusion_kwargs,
)
```

**Flow Matching 采样过程** (Euler 方法):

1. **初始化**: 从标准高斯分布采样初始噪声
   ```python
   x_0 = torch.randn(batch_size, *action_dims) * temperature
   ```

2. **迭代去噪** (默认 10 步):
   ```python
   for i in range(num_inference_steps):
       # 计算引导和无引导的速度场
       v_guided = step_fn(x_t, t)
       v_unguided = unguided_step_fn(x_t, t)

       # Classifier-Free Guidance
       v = (1 - w) * v_unguided + w * v_guided

       # Euler 积分
       x_{t+1} = x_t + dt * v
   ```

3. **输出**: 采样的动作序列 `[B * n_samples_total, T_f, C_action]`

**CFG 的作用**:
- **Guided**: 基于完整输入（包括导航信息）的预测
- **Unguided**: 基于无导航信息的预测
- **组合**: 通过权重 `inference_guidance_weight` 控制导航条件的影响强度

---

### 阶段 5: 动作转换为轨迹 (第 662-672 行)

**作用**: 将采样的动作序列转换为世界坐标系下的轨迹

```python
# 重复历史轨迹以匹配样本数
hist_xyz_rep = einops.repeat(
    ego_history_xyz[:, -1], "b ... -> (b n) ...", n=n_samples_total
)
hist_rot_rep = einops.repeat(
    ego_history_rot[:, -1], "b ... -> (b n) ...", n=n_samples_total
)

# 动作空间转换为轨迹
pred_xyz, pred_rot = self.action_space.action_to_traj(
    sampled_action,    # [B*n_samples, T_f, C_action]
    hist_xyz_rep,      # [B*n_samples, T_hist, 3]
    hist_rot_rep,      # [B*n_samples, T_hist, 4]
)
```

**动作空间说明**:
- 通常使用 `UnicycleAccelCurvature` 动作空间
- 动作维度: `C_action` 通常包括加速度和曲率
- 转换过程: 从动作积分得到未来轨迹的 XYZ 坐标和旋转

---

### 阶段 6: 重塑输出 (第 674-691 行)

**作用**: 将输出重塑为期望的格式

```python
# 重塑为 [B, num_traj_sets, num_traj_samples, T_f, 3/4]
pred_xyz = einops.rearrange(
    pred_xyz, "(b ns nj) ... -> b ns nj ...",
    ns=num_traj_sets,
    nj=num_traj_samples
)
pred_rot = einops.rearrange(
    pred_rot, "(b ns nj) ... -> b ns nj ...",
    ns=num_traj_sets,
    nj=num_traj_samples
)

# 可选：提取生成的文本 tokens
if kwargs.get("return_extra", False):
    extra = extract_text_tokens(self.tokenizer, vlm_outputs.sequences)
    # 重塑为 [B, num_traj_sets, num_traj_samples]
    for text_tokens in extra.keys():
        extra[text_tokens] = np.array(extra[text_tokens]).reshape(
            [input_ids.shape[0], num_traj_sets, num_traj_samples]
        )
    return pred_xyz, pred_rot, extra
```

---

## 三、输出数据 (Outputs)

### 主要输出

1. **`pred_xyz`**: `torch.Tensor [B, num_traj_sets, num_traj_samples, T_future, 3]`
   - 预测的未来轨迹 XYZ 坐标
   - T_future = 64（默认，6.4 秒，10Hz）

2. **`pred_rot`**: `torch.Tensor [B, num_traj_sets, num_traj_samples, T_future, 4]`
   - 预测的未来轨迹旋转（四元数）

### 可选输出（当 `return_extra=True`）

3. **`extra`**: `dict[str, np.ndarray]`
   - **`"cot"`**: Chain-of-Causation 推理文本
   - **`"meta_action"`**: 元动作描述
   - **`"answer"`**: VQA 回答（如果适用）
   - 每个字段的形状: `[B, num_traj_sets, num_traj_samples]`

---

## 四、推理流程总结

```
输入数据
  ├─ 多相机图像 (pixel_values)
  ├─ 历史轨迹 (ego_history_xyz/rot)
  ├─ 文本提示 (input_ids)
  └─ 导航信息（可选）
       ↓
[阶段 0] 数据预处理
  └─ 融合历史轨迹 tokens
       ↓
[阶段 1] VLM 自回归生成
  ├─ 视觉编码（多相机融合）
  ├─ 生成推理文本 (CoC)
  └─ 保存 KV 缓存（guided）
       ↓
[阶段 2] 构建无引导 KV 缓存
  ├─ 移除导航信息
  ├─ 预填充无引导前缀
  ├─ 重复 KV 缓存
  └─ 前向传播生成的 tokens
       ↓
[阶段 3] 定义去噪函数
  ├─ action_in_proj: 动作 → embeddings
  ├─ expert: Transformer 处理
  └─ action_out_proj: hidden → 动作
       ↓
[阶段 4] 扩散采样 (Flow Matching)
  ├─ 初始化噪声
  ├─ 迭代去噪（10 步，Euler）
  ├─ Classifier-Free Guidance
  │   ├─ v_guided (带导航)
  │   └─ v_unguided (无导航)
  └─ 采样动作序列
       ↓
[阶段 5] 动作转换为轨迹
  └─ action_to_traj: 从历史状态积分
       ↓
[阶段 6] 重塑输出
  └─ 重塑为 [B, sets, samples, T, dims]
       ↓
输出：轨迹预测 + 推理文本（可选）
```

---

## 五、关键技术要点

### 1. 混合架构
- **VLM**: 负责感知和推理（生成 CoC 文本）
- **Expert**: 负责轨迹预测（基于 VLM 的隐藏状态）
- **Diffusion**: Flow Matching 用于多模态轨迹采样

### 2. KV 缓存重用
- VLM 的 KV 缓存被 Expert 模型重用
- 通过交叉注意力机制连接两个模型
- 大幅减少计算量（避免重复编码）

### 3. Classifier-Free Guidance (CFG)
- 同时维护带条件和无条件的 KV 缓存
- 通过插值控制导航条件的影响强度
- 提高生成质量和可控性

### 4. 高效采样策略
- 视觉编码只运行一次（B 个样本）
- KV 缓存重复用于多个采样（B * n_samples）
- 避免重复的前向传播

### 5. 多样性采样
- 每个输入生成多个轨迹样本
- 捕捉未来的多模态性
- 支持下游任务的不确定性建模

---

## 六、性能考虑

### 内存优化
- 及时删除不需要的中间结果（如 logits）
- 使用 `torch.cuda.empty_cache()` 清理显存
- KV 缓存的批量重复而非完整复制

### 计算优化
- 使用 Flash Attention 2 加速注意力计算
- RoPE (Rotary Position Embedding) 用于位置编码
- 非因果注意力（可选）用于 Expert 模型

### 推理速度
- 默认配置下 H100 GPU:
  - `num_traj_samples=1`: ~24 GB VRAM
  - `num_traj_samples=16`: ~40 GB VRAM
  - `num_traj_samples=16` + CFG: ~60 GB VRAM

---

## 七、相关文件引用

- **主模型**: `src/alpamayo1_5/models/alpamayo1_5.py`
- **基础模型**: `src/alpamayo1_5/models/base_model.py`
- **动作投影**: `src/alpamayo1_5/models/action_in_proj.py`
- **扩散模型**: `src/alpamayo1_5/diffusion/flow_matching.py`
- **动作空间**: `src/alpamayo1_5/action_space/`
- **导航工具**: `src/alpamayo1_5/nav_utils.py`
