# 配置格式（schema_version=1）

必填字段：`experiment_key`、`name`、`program`。`args` 中的键会原样转换成
`--<key> <value>`；布尔值 `true` 转换为无值开关，`false`/`null` 被跳过。

```json
{
  "schema_version": 1,
  "experiment_key": "direction/model/variant/seed42/trial_001",
  "name": "human_readable_task",
  "program": "core_training/train_fusion.py",
  "gpu": 0,
  "env": {"PYTHONUNBUFFERED": "1"},
  "args": {"epochs": 3, "grad_checkpoint": 1},
  "passthrough": [],
  "required_paths": ["data/example.jsonl"],
  "log_path": "logs/<direction>/<model>/<variant>/seed42/trial_001/train.log",
  "resolved_config_path": "models/trained/<direction>/<model>/<variant>/seed42/trial_001/resolved_config.json"
}
```

目录和 `experiment_key` 统一采用：

```text
实验方向 / 模型或模型组合 / 方案 / seed / trial
```

`trial_001` 用于防止同一配置重复运行时覆盖，不承担实验分类作用。启动器仍兼容
带 `run_id` 的旧配置，但新配置不再使用它。

某个 CLI 参数需要按优先级选择已有 checkpoint 时，可写成：

```json
"ckpt": {
  "first_existing": ["adapter.pt.best_useful", "adapter.pt.best", "adapter.pt"]
}
```

同时把同一数组放进 `required_any_paths`，正式运行会在三者全都缺失时立即报错；
`--dry-run` 只提示尚未就绪，仍会打印优先级最高的预期命令。

`passthrough` 专供 `eval/run_segments_parallel.py`：启动器会先插入 `--`，再把该
数组原样传给三个 `eval_math.py` 子进程。
