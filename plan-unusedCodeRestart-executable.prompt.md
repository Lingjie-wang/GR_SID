# 执行版计划：Unused Code Restart（仅 RQ-VQ 路径）

## 0. 执行目标
在不改变当前默认行为的前提下，为 rqvae/rvq 路径增加 unused code restart 能力：
- 默认关闭（与当前行为一致）
- 可通过 Hydra 命令行覆盖开启
- 仅训练阶段生效，推理阶段无副作用

---

## 1. 变更范围（本次）
- 代码：src/modules/clustering/vector_quantization.py
- 配置：configs/experiment/rqvae_train_flat.yaml
- 配置：configs/experiment/rvq_train_flat.yaml

不在本次范围：
- MiniBatchKMeans 路径（rkmeans）
- 训练框架总流程改造
- 新增复杂调度策略（如 restart_interval）

---

## 2. 任务清单（按顺序执行）

### Task 1：参数接线（默认不改变行为）
目标：在 VectorQuantization 增加开关参数并保持默认关闭。

操作：
1. 修改 VectorQuantization.__init__，新增参数：
   - restart_unused_codes: bool = False
   - restart_noise_scale: float = 0.01
2. 将参数保存为实例属性。
3. 保持现有函数签名兼容，不改变已有调用方式。

完成标准：
- 不传新参数时，行为与当前一致。
- 代码可正常实例化。

---

### Task 2：实现 unused code 重启逻辑（最小侵入）
目标：当开启开关时，对当前 batch 未命中的 code 做重启。

操作：
1. 在 vector_quantization.py 增加私有方法：
   - _tile_with_noise(x, target_n, noise_scale)
   - _restart_unused_codes_if_needed(batch, assignments)
2. 逻辑规则：
   - 仅在 training 且 restart_unused_codes=True 时执行。
   - 用当前 batch 向量作为候选。
   - 若 batch_size < n_clusters，先重复并加噪扩充候选。
   - 找到本 step 未命中的 code（基于 assignments）。
   - 将未命中 code 的中心替换为随机候选向量。
3. 不修改 loss 接口，不改 quantization_strategy 接口。

完成标准：
- 开关关闭时，该逻辑不执行。
- 开关开启时，可实际替换未命中 code 的中心。

---

### Task 3：插入触发点
目标：把重启逻辑放在对训练影响最小且语义清晰的位置。

操作：
1. 在 VectorQuantization.model_step 内：
   - initialization 完成后
   - forward 拿到 assignments 后
   - 计算 loss 前
   插入 restart 调用。
2. 保证 predict/validation 路径不会触发训练态重启。

完成标准：
- 训练阶段可触发。
- 验证/推理阶段不触发。

---

### Task 4：配置暴露（显式默认 false）
目标：可通过配置和命令行控制开关，同时默认不变。

操作：
1. 在 rqvae_train_flat.yaml 的 model.quantization_layer 节点新增：
   - restart_unused_codes: false
   - restart_noise_scale: 0.01
2. 在 rvq_train_flat.yaml 的 model.quantization_layer 节点同样新增。

完成标准：
- 两个实验配置都显式声明默认 false。
- 旧命令无需改动即可运行。

---

### Task 5：轻量可观测性
目标：可确认机制是否生效，但不污染日志。

操作：
1. 记录本 step 重启数量（如 restart_count）。
2. 仅在 verbose 模式下输出简洁日志。

完成标准：
- 开关开启且存在未命中 code 时，可观察到 restart_count > 0。
- 日志量可控。

---

## 3. 回归验证清单

### 验证 A：配置默认值
命令：
python -m src.train experiment=rqvae_train_flat --cfg job
python -m src.train experiment=rvq_train_flat --cfg job

检查点：
- model.quantization_layer.restart_unused_codes 为 false。

---

### 验证 B：默认行为回归（不开开关）
命令（示例，短跑）：
python -m src.train experiment=rqvae_train_flat trainer.max_steps=20
python -m src.train experiment=rvq_train_flat trainer.max_steps=20

检查点：
- 训练不报错。
- 指标量级与当前基线一致（允许正常抖动）。

---

### 验证 C：开关生效
命令（示例，短跑）：
python -m src.train experiment=rqvae_train_flat trainer.max_steps=20 model.quantization_layer.restart_unused_codes=true
python -m src.train experiment=rvq_train_flat trainer.max_steps=20 model.quantization_layer.restart_unused_codes=true

检查点：
- 训练不报错。
- 可观测到 restart_count（或等效日志）出现。

---

### 验证 D：推理无副作用
命令（按你现有推理命令模板）：
python -m src.inference experiment=rqvae_inference_flat
python -m src.inference experiment=rvq_inference_flat

检查点：
- 推理流程正常。
- 不触发训练态 restart 逻辑。

---

## 4. 风险与保护
- 风险：batch 很小时候选不足。
  - 保护：tile + noise 扩充。
- 风险：重启过于频繁导致波动。
  - 保护：先不加复杂策略，首版保持最小实现；通过实验观察决定是否加 restart_interval。
- 风险：无意改变旧行为。
  - 保护：代码默认 false + 配置显式 false + 默认回归测试。

---

## 5. 交付定义（DoD）
以下全部满足即完成：
1. 三个文件改动完成且通过基本运行。
2. 默认不开开关时，行为与当前版本一致。
3. 开启开关时，unused code restart 可触发且训练稳定。
4. 推理路径无副作用。
5. 变更说明可复现（含命令与检查点）。

---

## 6. 建议提交信息（可选）
feat(rqvq): add optional unused-code restart for VectorQuantization with safe default off
