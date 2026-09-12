# BES 任务感知接入位置实验方案

## 目标与判定原则

目标不是寻找 CKA 最高的位置，而是在相同训练 token、可训练参数与评测预算下，找到能稳定改善下游任务且计算成本合理的 `(large_start, large_end, small_start, small_end)`。

- 小模型区间在筛选结果与汇总文件中统一写作右开区间 `[a,b)`。
- `train_fusion.py` 为兼容已有 checkpoint，命令行 `--small_end` 仍是 inclusive；实验工具会自动传入 `b-1`。
- CKA、`E_in`、`E_seg` 仅作为诊断。最终排名由独立题目的真实 `ΔNLL`、等预算短训和多 seed 任务指标决定。
- 差异落在不确定区间内时判为平局，并选择 FLOPs 更低的配置。

## 先做四个必要对照

在完整搜索前，对默认 1/3→2/3 配置比较：

1. 当前预训练小模型片段。
2. identity control：跳过小模型层，只保留同容量 bridge。
3. 随机初始化后冻结的小模型片段。
4. 预训练片段加小模型 LoRA。

如果预训练片段不能跨 seed 稳定超过 identity/random control，说明收益主要来自 bridge，此时不应继续精细选层；应先降低 bridge 容量或给小模型明确的局部蒸馏目标。

默认位置的 bridge-only control 可直接运行：

```bash
python3 core_training/train_fusion.py --data data/mix_all.jsonl \
  --bypass_small --contrast_weight 0 --seed 42 \
  --out cache/control_bridge_only_seed42.pt

python3 eval/eval_math.py --bench segments --limit 300 \
  --bypass_small --ckpt cache/control_bridge_only_seed42.pt.best \
  --tag control_bridge_only_seed42
```

control 与正式候选必须使用相同的 bridge 维度、训练 token、优化器、位置和随机种子。

## 三阶段搜索

运行：

```bash
bash scripts/run_task_aware_search.sh
```

可通过环境变量覆盖服务器设置：

```bash
GPUS=0,1 PARALLEL=2 STAGE1_GPU=1 DATA=data/mix_all.jsonl \
  bash scripts/run_task_aware_search.sh
```

阶段 1 对大小模型深度、片段长度和捕获→注入跨度做分层采样。每个候选先拟合低成本线性输出 probe，再把更新缩放到残差 RMS 的 1%、3%、10%，真正运行完整模型并测量配对 `ΔNLL`。同时报告 identity control，用于判断小模型层提供了多少新增任务信息。

阶段 2 对前 8 个候选和默认方案做 3 个训练 seed 的短训。位置比较默认关闭 contrast loss，主指标是独立验证集 `baseline - fusion` CE gain。

阶段 3 对前 3 个候选和默认方案提高训练预算，并用 3 个不同题目采样 seed 评测 GSM8K、MATH 1-3、MATH 4-5。聚合时直接使用正确题数，不使用四舍五入后的百分比。

## 最终确认

对 `cache/task_aware_search/finalists.json` 中两个候选及默认方案：

- 每个配置至少训练 5 个 seed。
- 保存逐题预测，在完全相同题目上做分层 paired bootstrap 与 McNemar 检验。
- 同时报 response NLL、三类准确率、branch/残差 RMS、gate、训练时间、推理延迟和显存。
- 最终测试集不得参与 checkpoint、位置或超参数选择。
- 若准确率差异的 95% CI 包含 0，则结论写作“不可区分”，按推理 FLOPs 选择较短片段。

两个候选在同一题集上的配对检验：

```bash
python3 scripts/compare_position_predictions.py \
  --a eval_candidate_A.json --b eval_candidate_B.json \
  --label fusion --bootstrap 10000
```

## 推荐消融顺序

1. 默认、单点前段、单点后段、双点。
2. `S(x)` 与 `S(x)-x`。
3. bridge-only、随机片段、预训练片段。
4. 冻结小模型、小模型 LoRA。
5. 只有小模型已显示稳定价值后，再比较大模型接收端 LoRA 或逐步替换原 MLP。
