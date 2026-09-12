param(
    [string]$Python = "$env:USERPROFILE\.conda\envs\BES\python.exe",
    [string]$OutDir = "cache\local_smoke",
    [int]$MaxLen = 64,
    [int]$MaxNew = 8,
    [int]$BridgeMlpDim = 128,
    [int]$LoraRank = 4
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $RepoRoot

if (-not (Test-Path -LiteralPath $Python)) {
    throw "BES Python not found: $Python"
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $OutDir "eval") | Out-Null

# Model files are expected in the local ModelScope/Hugging Face cache.  This
# prevents an accidental multi-GB model download during a smoke test.
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

function Invoke-BesPython {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)
    & $Python @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed ($LASTEXITCODE): $Python $($Args -join ' ')"
    }
}

Write-Host "[0/8] Environment and regression tests"
Invoke-BesPython -c "import torch, transformers, peft; print(torch.__version__, transformers.__version__, peft.__version__); print(torch.cuda.get_device_name(0))"
Invoke-BesPython test\test_position_selection.py -v

Write-Host "[1/8] Tiny task-aware position screening"
Invoke-BesPython core_training\select_positions.py `
    --data data\mix_all.jsonl --num_samples 6 --max_len 32 --batch_size 1 `
    --fit_ratio 0.5 --topk 1 --lam 0.01 --inject_spans "1,2" `
    --segment_lengths "1,2" --exit_topk 2 --exit_samples 2 --exit_max_len 32 `
    --intervention_ratios 0.05 --rank_ratio 0.05 `
    --out (Join-Path $OutDir "position_selection.json")

Write-Host "[2/8] Deep bridge, both base models frozen"
Invoke-BesPython core_training\train_fusion.py `
    --data data\mix_all.jsonl --max_samples 1 --epochs 1 --batch_size 1 `
    --max_len $MaxLen --bridge_depth 2 --bridge_mlp_dim $BridgeMlpDim `
    --small_lora_r 0 --grad_checkpoint 1 --attn_impl sdpa --warmup_steps 0 `
    --eval_samples 1 --eval_batch_size 1 --eval_max_samples 1 --eval_every 1 `
    --contrast_weight 0 --log_every 1 `
    --out (Join-Path $OutDir "deep_bridge.pt") --plot ""

Write-Host "[3/8] Small-model segment LoRA plus bridge"
Invoke-BesPython core_training\train_fusion.py `
    --data data\mix_all.jsonl --max_samples 1 --epochs 1 --batch_size 1 `
    --max_len $MaxLen --bridge_depth 1 --bridge_mlp_dim $BridgeMlpDim `
    --small_lora_r $LoraRank --small_lora_alpha (2 * $LoraRank) --small_lora_dropout 0 `
    --grad_checkpoint 1 --attn_impl sdpa --warmup_steps 0 `
    --eval_samples 1 --eval_batch_size 1 --eval_max_samples 1 --eval_every 1 `
    --contrast_weight 0 --log_every 1 `
    --out (Join-Path $OutDir "small_lora_bridge.pt") --plot ""

Write-Host "[4/8] True 4B LoRA baseline"
Invoke-BesPython core_training\train_lora.py `
    --data data\mix_all.jsonl --max_samples 1 --epochs 1 --batch_size 1 `
    --max_len $MaxLen --grad_checkpoint 1 --attn_impl sdpa `
    --lora_r $LoraRank --lora_alpha (2 * $LoraRank) --lora_dropout 0 `
    --warmup_steps 0 --eval_samples 1 --eval_batch_size 1 --eval_max_samples 1 `
    --eval_every 1 --log_every 1 `
    --out (Join-Path $OutDir "large_lora_r$LoraRank") --plot ""

$EvalDir = Join-Path $OutDir "eval"

Write-Host "[5/8] Reload and evaluate deep bridge"
Invoke-BesPython eval\eval_math.py --bench math --limit 1 --max_new $MaxNew `
    --ckpt (Join-Path $OutDir "deep_bridge.pt.best") `
    --bridge_depth 2 --bridge_mlp_dim $BridgeMlpDim `
    --out_dir $EvalDir --tag deep_bridge

Write-Host "[6/8] Reload and evaluate small-model LoRA plus bridge"
Invoke-BesPython eval\eval_math.py --bench math --limit 1 --max_new $MaxNew `
    --ckpt (Join-Path $OutDir "small_lora_bridge.pt.best") `
    --small_lora_ckpt (Join-Path $OutDir "small_lora_bridge.pt.best.small_lora") `
    --bridge_depth 1 --bridge_mlp_dim $BridgeMlpDim `
    --out_dir $EvalDir --tag small_lora_bridge

Write-Host "[7/8] Reload and evaluate true 4B LoRA"
Invoke-BesPython eval\eval_math.py --bench math --limit 1 --max_new $MaxNew `
    --lora_ckpt (Join-Path $OutDir "large_lora_r$LoraRank.best") `
    --out_dir $EvalDir --tag "large_lora_r$LoraRank"

Write-Host "[8/8] Previous three-segment evaluation path"
Invoke-BesPython -c "import pyarrow"
Invoke-BesPython eval\eval_math.py --bench segments --limit 1 --max_new $MaxNew `
    --ckpt (Join-Path $OutDir "deep_bridge.pt.best") `
    --bridge_depth 2 --bridge_mlp_dim $BridgeMlpDim `
    --out_dir $EvalDir --tag segments_smoke

Write-Host "Local smoke completed. Outputs: $OutDir"
