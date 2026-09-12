# Bridge 容量、小模型 LoRA 与大模型 LoRA 实验

## 参数量设计

三组实验不做强制等参数，只保持数据、训练 token、优化器和评测设置一致。
正式归因时应加 `RUN_CONTROLS=1`，额外训练冻结小模型标准 bridge，以及 depth1/depth2
同容量 bridge-only 对照；默认关闭是为了避免常规三组流程的训练时长翻倍。

| 实验 | 可训练部分 | 默认配置 | 预计可训练参数 |
| --- | --- | --- | ---: |
| 深 bridge | 输入/输出 bridge；大小模型冻结 | depth=2, MLP=4096 | 约 88.1M |
| 小模型 LoRA | depth=1 bridge + 实际使用的小模型片段 | bridge 44.1M + LoRA r=32 | 约 51–52M |
| 4B LoRA baseline | 4B 全部 attention/MLP 线性层 | r=64, alpha=128 | 约 132M |

实际参数量由训练脚本根据模型结构打印，应以运行日志为准。

深 bridge 不是简单扩大宽度，而是在 adapter1 和 adapter2 的输出空间各增加一个预归一化残差 GLU block。额外 block 使用可学习残差门控，避免直接堆叠导致训练初期更新过大。

小模型 LoRA 只注入当前旁路真正运行的 `[small_start, small_end]` 层，目标模块为 `q/k/v/o/gate/up/down_proj`。默认片段约十层时，r=32 通常增加约 7–8M 参数；片段长度改变时参数量也会线性变化。

大模型 baseline 使用全 36 层、全部七类线性模块和 r=64。这是一组常规强 LoRA baseline，而不是为了匹配 bridge 的 44M 将 rank 人为压低到 21。

## 运行

推荐使用两张卡分两阶段运行。GPU 0 只筛选位置；完成后 GPU 1 串行训练三组实验，
并在 GSM8K、MATH Level 1–3、MATH Level 4–5 上各评测 400 题：

```bash
SELECTION_GPU=0 EXPERIMENT_GPU=1 bash scripts/run_two_gpu_pipeline.sh
```

也可以拆开提交两个作业：

```bash
STAGE=selection SELECTION_GPU=0 bash scripts/run_two_gpu_pipeline.sh
STAGE=experiments EXPERIMENT_GPU=1 PREPARE_DATA=0 bash scripts/run_two_gpu_pipeline.sh
```

第二阶段自动采用筛选 JSON 中排名第一的位置。训练数据默认约 3.2 万条：约 2.4 万
MATH（官方 train + MetaMathQA-MATH）和 8000 GSM8K；官方 MATH train 的 Level 1–5
全部覆盖，并保留分层验证集。构造时会去重并封锁官方 MATH test 的精确题面。
服务器正式默认值按 80GB GPU 设置为 batch size 8、训练长度 1536、生成长度 1024、
3 epochs、warmup 400 steps、每 500 steps 验证、FlashAttention 2 和 seed 42。
旁路优化使用普通 response-only CE，并另设 400-step 固定 `branch_alpha=0.05` 的启动阶段，随后释放 alpha；前
200 step 只训练 bridge，之后才解冻小模型 LoRA。bridge、小模型 LoRA、alpha 的
学习率分别为 `1e-4 / 2e-5 / 5e-4`。baseline 重加权和旧 JS 都默认关闭；zero 与
shuffled 只在验证时作为反事实诊断，不参与反向传播。小模型 LoRA dropout 在该实验
中设为 0，避免 correct/shuffled 对照混入不同 dropout 掩码造成的随机差异；大模型
LoRA baseline 仍保留 0.05 dropout。

仅运行三组容量实验：

```bash
cd BES
GPU=0 bash scripts/run_capacity_lora_experiments.sh
```

覆盖训练预算：

```bash
GPU=0 MAX_SAMPLES=10000 EPOCHS=3 BATCH_SIZE=8 MAX_LEN=1024 \
  ACC_LIMIT=400 SEED=42 bash scripts/run_capacity_lora_experiments.sh
```

使用已经筛选的位置。注意 `SMALL_END` 是训练命令的 inclusive 终点：

```bash
LARGE_START=17 LARGE_END=29 SMALL_START=3 SMALL_END=3 \
  GPU=0 bash scripts/run_capacity_lora_experiments.sh
```

只训练、不运行生成评测：

```bash
RUN_EVAL=0 GPU=0 bash scripts/run_capacity_lora_experiments.sh
```

## 输出

- `deep_bridge_seed*.pt`：深 bridge 权重。
- `small_lora_seed*.pt`：bridge 权重。
- `small_lora_seed*.pt.small_lora/`：与 bridge 配套的小模型 LoRA。
- `large_lora_r64_seed*/`：纯 4B LoRA adapter。
- `eval/`：三组实验与共享纯 4B baseline 的 GSM8K、MATH low、MATH high 逐题结果。

建议至少运行 seed 42、43、44。单个 seed 只能用于排查训练是否正常，不能用于判断三种方法的优劣。
