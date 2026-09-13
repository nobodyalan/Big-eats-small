# BES 实验配置

`configs/` 只保存体积很小、可复现且应该进入 Git 的实验定义。模型权重、日志和
评测结果分别写入被 Git 忽略的 `models/`、`logs/` 和 `eval_results/`。

## 目录约定

```text
configs/
├── datasets/                 # 数据配方与固定切分说明
├── models/                   # 基础模型身份、结构和 tokenizer 约束
└── experiments/              # 实验方向/模型组合/方案/训练或评测配置
    ├── large_lora/qwen3_4b/r48/
    └── single_bridge/qwen3_4b__qwen3_06b/<position>/

models/
├── pretrained/               # 可选的本地基础模型/软链接
└── trained/<direction>/<model>/<variant>/<seed>/<trial>/

logs/<direction>/<model>/<variant>/<seed>/<trial>/
eval_results/<direction>/<model>/<variant>/<seed>/<trial>/
cache/                        # 可重建的下载、隐状态、搜索和临时缓存
```

分类顺序固定为“实验方向 / 模型或模型组合 / 方案 / seed / trial”。
`experiment_key` 使用相同的语义路径，因此不需要记忆日期型 `run_id`。同一配置
再次运行时递增 `trial_001`，防止覆盖。不要为单次实验新增 shell 脚本；复制对应
JSON 配置并修改方案、seed 或 trial 即可。

## 运行

先检查最终命令和路径：

```bash
python3 scripts/run_config.py \
  configs/experiments/large_lora/qwen3_4b/r48/train_seed42_trial001.json \
  --dry-run
```

服务器后台启动（配置会自行把子进程日志追加到 `log_path`）：

```bash
nohup python3 scripts/run_config.py \
  configs/experiments/large_lora/qwen3_4b/r48/train_seed42_trial001.json \
  > /dev/null 2>&1 &
echo $!
```

调试时需要在终端看到实时输出，可加 `--no-log`。正式运行会在模型目录保存
`resolved_config.json`，其中包含实际命令、时间和 Git commit。

`small_end` 仍遵循当前 `train_fusion.py` 的 inclusive 语义；文档里写成 `[a,b)`
时，配置里的 `small_end` 应填 `b-1`。

## 保留策略

- LoRA：保留 `.best`，最终目录仅在需要恢复训练时保留。
- Bridge：优先保留 `.best_useful`，否则保留 `.best`。
- `eval_results/` 只放 JSON/汇总；逐题输出和训练曲线放 `logs/`。
- `cache/` 中的内容应可删除重建，不能作为唯一实验记录。
