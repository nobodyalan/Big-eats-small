# 门控残差融合（Gated Residual Fusion）

在 Qwen3-4B 前向的 1/3~2/3 层之间插入一条由 Qwen3-0.6B 中段层构成的可训练旁路：

```
第 12 层输出 → adapter1(2560→1024) → 0.6B 第 9~18 层 → adapter2(1024→2560) → ×gate → 加回第 24 层残差流
```

冻结两个大模型，只训练约 **22M** 旁路参数。初始 `gate≈0` + `adapter2` 零初始化保证旁路输出恒为 0（不破坏原 4B）。

## 目录结构

```
BES/
├── core_training/          # 核心架构与训练
│   ├── main.py               # 融合架构 + 推理入口
│   └── train_fusion.py       # 适配器训练脚本
├── data/                   # 数据集（大文件, 不入库, 需自行准备）
│   ├── metamath_gsm8k_zh.json   # MetaMathQA_GSM8K_zh 原始数据
│   └── train_metamath.jsonl     # 训练数据（由 scripts 生成）
├── test/                   # 测试脚本
│   ├── test_fusion.py         # 冒烟测试（恒等性 + 解码）
│   └── test_baseline.py       # 基线评测（纯 4B / 0.6B 文本生成）
├── eval/                   # 评测
│   └── eval_questions.py      # 评测题集
├── scripts/                # 工具脚本
│   └── convert_metamath.py    # MetaMathQA → 训练 jsonl 转换
├── cache/                  # 训练产物（权重 / loss 图, 不入库）
├── setup_math_train.ipynb  # 远程 JupyterLab 跑 MATH 训练的 notebook
└── README.md
```

## 环境安装

```bash
conda create -n bes python=3.11 -y
conda activate bes
pip install torch --index-url https://download.pytorch.org/whl/cu121   # 按服务器 CUDA 版本选
pip install transformers accelerate modelscope matplotlib datasets
```

## 快速开始（所有命令在 BES 根目录执行）

```bash
# 1. 冒烟测试：验证架构（旁路初始输出应为 0.0）
python test/test_fusion.py

# 2. 训练适配器（默认用 data/train_metamath.jsonl）
python core_training/train_fusion.py --max_samples 2000 --epochs 1

# 3. 融合推理
python core_training/main.py

# 4. 基线评测（对照组）
python test/test_baseline.py

# 5. 任务感知接入位置搜索（分层候选 → ΔNLL → 多 seed 短训）
bash scripts/run_task_aware_search.sh

# 只做位置筛选，不启动短训和准确率评测
bash scripts/run_selection_only.sh

# 深 bridge / 小模型 LoRA / 标准 4B LoRA 三组实验
bash scripts/run_capacity_lora_experiments.sh

# 两阶段正式流程：GPU0 筛位置；GPU1 训练三组并做三段各 400 题评测
SELECTION_GPU=0 EXPERIMENT_GPU=1 bash scripts/run_two_gpu_pipeline.sh

# Windows + BES conda + 12GB GPU：串行验证筛选、三组训练及分层评测基本流程
powershell -ExecutionPolicy Bypass -File scripts/run_local_smoke.ps1
```

本地 smoke 默认只训练 1 条、评测每段 1 题，并使用窄 bridge 与 LoRA r=4；它用于发现
依赖、显存、保存/重载和参数连接问题，输出准确率不具有统计意义。正式实验仍使用
`run_capacity_lora_experiments.sh` 中的 bridge 4096 与大模型 LoRA r=64 配置。

两阶段流程会先生成约 75% MATH、25% GSM8K 的训练/验证集；MATH 官方 train
按 Level 1–5 分层切分，官方 test 只用于最终评测。可分别执行
`STAGE=selection` 和 `STAGE=experiments`，第二阶段会自动读取第一阶段的最佳位置。
正式脚本默认按 80GB GPU 使用 batch size 8、max length 1024、3 epochs、
FlashAttention 2、生成长度 1024 和 seed 42。

任务感知搜索的指标、预算和最终统计规则见
[`TASK_AWARE_EXPERIMENT_PLAN.md`](TASK_AWARE_EXPERIMENT_PLAN.md)。筛选结果里的小模型片段统一表示为
右开区间 `[a,b)`；兼容旧训练入口时，脚本会自动把它转换为 inclusive `--small_end=b-1`。

## 数据准备

```bash
# 从原始 MetaMathQA json 生成训练 jsonl（默认输出 data/train_metamath.jsonl）
python scripts/convert_metamath.py

# MATH 数据集训练：见 setup_math_train.ipynb
# （在远程 JupyterLab 上：下载 Hendrycks MATH → 转 jsonl → 跑训练）
```

> 训练只用 train 划分；MATH 的 test 划分（5k）留作独立评测，避免数据泄漏。

## 训练关键参数（`train_fusion.py`）

| 参数 | 默认 | 说明 |
|------|------|------|
| `--data` | `data/mix_all.jsonl` | 训练数据（prepare_data.py 生成，验证集在末尾） |
| `--max_samples` | 0 | 最多训练条数（0=全部；评估集另算，不被截断） |
| `--max_len` | 1024 | 单条最大 token 数（显存不足调小） |
| `--lr` | 1e-4 | 学习率 |
| `--batch_size` | 8 | batch（H100 建议 8~16；12GB 用 1） |
| `--epochs` | 3 | 训练轮数 |
| `--eval_every` | 200 | 每隔 N 步评估 fusion / baseline loss |
| `--eval_samples` | 64 | 从数据末尾留出多少条作评估集 |
| `--eval_batch_size` | 8 | 评估时的 batch（越大评估越快） |
| `--eval_max_samples` | 400 | 每次评估最多用多少条（验证集大时设小，0=全部） |
| `--patience` | 0 | 早停耐心（0=关闭） |
| `--contrast_weight` | 0 | InterLat 式 JS 对比损失权重（防旁路塌缩，需 `--batch_size ≥ 2`） |

## 输出

- `cache/fusion_adapter.pt` — 最终权重
- `cache/fusion_adapter.pt.best` — 最优权重（早停时）
- `cache/train_fusion_loss.png` — loss 图
- `eval_results/` — 评测结果（`main.py` / `test_baseline.py` 生成）

## 实现要点

- 适配器用 **fp32 主权重 + bf16 前向**（混合精度），避免 bf16 参数更新冻结。
- 训练时两个 `up` 投影重初始化为小随机值（打破零初始化梯度锁）。
- 梯度检查点省显存，可在 12GB 显存上跑。
