# -*- coding: utf-8 -*-
"""
门控残差融合(Gated Residual Fusion): 在 4B 前向的 1/3~2/3 层之间插入一条
由 0.6B 中段层构成的可训练旁路。

整体流程:
  1. 加载 4B(主脑)与 0.6B(旁路中段层的宿主,冻结)。
  2. attach_fusion 用前向钩子把门控残差旁路挂到 4B 上:
     第 pos1 层输出 → adapter1(2560→1024) → 0.6B 中段层 → adapter2(1024→2560)
     → ×gate → 加回第 pos2 层残差流。
  3. 4B 正常前向 + 文本解码(旁路可训练,初始 gate≈0 严格保持原 4B 行为)。

依赖: torch / transformers / modelscope(自动定位模型本地路径)
运行: python main.py
"""

import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# 项目根目录 = 上一级(BES); main 在 core_training/ 下, 需把 eval/ 加入导入路径
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "eval"))

from eval_questions import EVAL_QUESTIONS  # 推理能力测试题集(评测模式使用)

# Windows 控制台默认 GBK 编码,打印中文可能报 UnicodeEncodeError,统一转为 UTF-8 输出
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  ① 超参数配置 —— 所有超参数统一放在此处,只需改这里                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝
@dataclass
class Config:
    # ── 模型 ──────────────────────────────────────────────────────────────
    model_small_id: str = "Qwen/Qwen3-0.6B"                # 旁路中段层的宿主(0.6B,冻结)。
                                                        # 注意: 官方没有 Qwen3-0.6B-Instruct(Instruct 从 1.7B 起),此处只能用基座版
    model_large_id: str = "Qwen/Qwen3-4B-Instruct-2507"    # 主脑(深度推理 + 最终解码)
    model_small_local: Optional[str] = None                # 若已下载可填本地路径,否则留 None
    model_large_local: Optional[str] = None
    dtype: str = "auto"        # 精度: "auto"(有 GPU 用 bf16,纯 CPU 用 float32) / "bfloat16" / "float32"
    device_map: str = "auto"   # 设备映射: "auto" / "cpu" / "cuda:0"

    # ── 门控残差融合架构 ────────────────────────────────────────────────────
    fusion_enabled: bool = True       # 在 4B 的 1/3 与 2/3 层之间加"门控残差旁路":
                                      #   h(1/3处) → adapter1(2560→1024) → 0.6B 中段层(1/3~2/3)
                                      #   → adapter2(1024→2560) → ×gate → 加回 2/3 处残差流
    fusion_pos1_frac: float = 1 / 3   # 4B 取隐状态的位置(36 层 → 第 12 层输出)
    fusion_pos2_frac: float = 2 / 3   # 4B 加回残差的位置(第 24 层输出)
    fusion_mlp_dim: int = 4096        # 适配器 MLP 中间层维度(可调; 4096 → 约 44M 旁路参数)
    fusion_bridge_depth: int = 1      # 每个 adapter 的深度；2 会增加一个同输出维残差 GLU block
    # 4B 接入位置(取隐状态层/加回残差层, 0-based); None = 自动用 1/3 / 2/3 位置
    fusion_large_start: Optional[int] = None
    fusion_large_end: Optional[int] = None
    # 0.6B 旁路层范围(含端点, 0-based); None = 自动用 1/3~2/3 位置。负索引从末尾数。
    # 例: start=0, end=-1 → 从第一个隐藏层接到最后一个隐藏层(整段 0.6B)
    fusion_small_start: Optional[int] = None
    fusion_small_end: Optional[int] = None
    fusion_bypass_small: bool = False  # True=跳过冻结小模型层，作为 bridge-only 对照

    # ── 最终解码 ──────────────────────────────────────────────────────────
    max_new_tokens: int = 512
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20                 # 采样 top-k(Qwen3 官方推荐参数 TopK=20,抑制发散;0=关闭)
    repetition_penalty: float = 1.0   # 解码重复惩罚(>1 抑制已出现 token, 打断"英镑英镑..."式循环)

    # ── 其他 ──────────────────────────────────────────────────────────────
    cache_dir: str = "cache"    # 磁盘缓存目录(其他脚本如 bridge_train.py 会用到)
    prompt: str = "请一步一步思考:一个水龙头每分钟流出 12 升水,一个水箱容量是 300 升,请问注满水箱需要多少分钟?"
    seed: int = 42              # 随机种子(解码采样可复现)
    device_map_small: Optional[str] = None   # 显存不足时可将 0.6B 单独放 CPU: 填 "cpu"(默认跟随 device_map)

    # ── 评测模式 ──────────────────────────────────────────────────────────
    enable_eval: bool = False   # True = 对 eval_questions.py 的测试题集逐题跑融合管线,
                                #        结果保存到 eval_results/fused_<时间戳>.json
                                # False = 只跑上面单条 prompt 演示


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  ② 基础小工具                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def resolve_dtype(dtype_str: str) -> torch.dtype:
    """把配置字符串转成 torch 精度;auto 时按是否有 GPU 自动选择"""
    if dtype_str == "auto":
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    return getattr(torch, dtype_str)


def model_device(model) -> torch.device:
    """取模型实际所在设备(兼容 device_map="auto")"""
    return next(model.parameters()).device


def resolve_model_path(model_id: str, local_override: Optional[str]) -> str:
    """优先使用完整的本地缓存，只有本地没有模型时才请求 ModelScope。"""
    if local_override:
        return local_override

    def is_complete_model_dir(path: str) -> bool:
        if not os.path.isfile(os.path.join(path, "config.json")):
            return False
        weight_names = (
            "model.safetensors", "model.safetensors.index.json",
            "pytorch_model.bin", "pytorch_model.bin.index.json",
        )
        return any(os.path.isfile(os.path.join(path, name)) for name in weight_names)

    # ModelScope 新版缓存。snapshot_download 在离线环境仍可能先请求文件列表，
    # 因此要在调用它之前直接解析已经完整落盘的 snapshot。
    cache_root = os.environ.get(
        "MODELSCOPE_CACHE",
        os.path.join(os.path.expanduser("~"), ".cache", "modelscope"),
    )
    model_dir_name = model_id.replace("/", "--")
    ms_candidates = [
        os.path.join(cache_root, "models", model_dir_name, "snapshots", "master"),
        os.path.join(cache_root, "hub", model_id),
        os.path.join(cache_root, "hub", "models", model_id),
    ]
    for path in ms_candidates:
        if is_complete_model_dir(path):
            return path

    # Hugging Face 的 snapshots 目录可能只缓存了 config；仅选择同时含权重的版本。
    hf_model_dir = os.path.join(
        os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface")),
        "hub", f"models--{model_dir_name}", "snapshots",
    )
    if os.path.isdir(hf_model_dir):
        snapshots = sorted(
            (os.path.join(hf_model_dir, name) for name in os.listdir(hf_model_dir)),
            key=os.path.getmtime,
            reverse=True,
        )
        for path in snapshots:
            if is_complete_model_dir(path):
                return path

    offline = any(
        os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "MODELSCOPE_OFFLINE")
    )
    if offline:
        # 交给 Transformers 的 local-files-only 语义报告明确的缓存缺失错误，
        # 不能在调用方明确要求离线时偷偷转去 ModelScope 联网。
        return model_id

    try:
        from modelscope import snapshot_download
        return snapshot_download(model_id)
    except ImportError:
        return model_id  # 没有 modelscope 时直接用 HuggingFace 模型 id


def sample_token(logits: torch.Tensor, temperature: float, top_p: float,
                 seen_ids=None, repetition_penalty: float = 1.0, top_k: int = 0) -> torch.Tensor:
    """带 temperature / top_p / top_k / 重复惩罚的采样,返回形状 (1, 1) 的 token id"""
    logits = logits.float().clone()
    if seen_ids and repetition_penalty != 1.0:
        for tid in seen_ids:                # 压低已生成 token 的分数,打断重复循环
            if repetition_penalty > 1.0:
                logits[0, tid] /= repetition_penalty
            else:
                logits[0, tid] *= repetition_penalty
    if top_k > 0:                           # top-k 过滤: 只保留概率最大的 k 个候选,抑制发散
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
        logits[logits < v[..., -1:]] = float("-inf")
    logits = logits / max(temperature, 1e-6)
    probs = torch.softmax(logits, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        mask = cum > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False  # 至少保留概率最大的一个 token
        probs = torch.zeros_like(probs).scatter_(-1, sorted_idx,
                                                 sorted_probs.masked_fill(mask, 0.0))
        probs = probs / probs.sum(dim=-1, keepdim=True)
    return torch.multinomial(probs, 1)


def build_prompt_text(tokenizer, prompt: str) -> str:
    """构造对话模板文本;0.6B(基座)与 4B(Instruct)共用同一分词器与模板"""
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
    return prompt


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  ⑦ 主脑 4B: 文本解码                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def decode_answer(model, tokenizer, h_last: torch.Tensor, past, config: Config) -> str:
    """从 4B 最后的隐状态出发,逐 token 采样解码成文本"""
    dev = model_device(model)
    eos = tokenizer.eos_token_id
    generated = []
    # 与 generate 一致: 每步给"全长"attention mask(缓存 + 当前),保证位置编码正确
    attention_mask = torch.ones((1, past.get_seq_length()), device=dev)
    with torch.inference_mode():
        for _ in range(config.max_new_tokens):
            logits = model.lm_head(h_last)                       # (1, 1, V)
            next_token = sample_token(logits[:, -1, :], config.temperature, config.top_p,
                                      seen_ids=generated,
                                      repetition_penalty=config.repetition_penalty,
                                      top_k=config.top_k)
            tok = int(next_token.item())
            generated.append(tok)
            if tok == eos:
                break
            next_emb = model.model.embed_tokens(next_token)      # 真实 token 走正常 embedding,不再对齐
            attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=dev)], dim=1)
            outputs = model.model(
                inputs_embeds=next_emb,
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
            )
            past = outputs.past_key_values
            h_last = outputs.last_hidden_state[:, -1:, :]
    return tokenizer.decode(generated, skip_special_tokens=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  ⑦.5 门控残差融合架构(新方案)                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝
class GatedAdapter(nn.Module):
    """
    门控 + 投影适配器(GLU 风格):
      out = up( act(down(x)) × sigmoid(gate(x)) )
    up 投影零初始化 → 初始输出恒为 0,保证"初始门控保持相同输入"。
    MLP 维度(mlp_dim)可调。
    """

    def __init__(self, in_dim: int, out_dim: int, mlp_dim: int, depth: int = 1):
        super().__init__()
        if depth < 1:
            raise ValueError("bridge depth 必须 >= 1")
        self.down = nn.Linear(in_dim, mlp_dim)
        self.gate = nn.Linear(in_dim, mlp_dim)
        self.act = nn.SiLU()
        self.up = nn.Linear(mlp_dim, out_dim)
        self.blocks = nn.ModuleList(
            GatedResidualBlock(out_dim, mlp_dim) for _ in range(depth - 1))
        # 零初始化 up 投影: 无论 down/gate 输出什么,适配器初始输出都是 0
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        d = self.act(self.down(x))
        g = torch.sigmoid(self.gate(x))
        x = self.up(d * g)
        for block in self.blocks:
            x = block(x)
        return x


class GatedResidualBlock(nn.Module):
    """用于加深 bridge 的预归一化残差 GLU block。"""

    def __init__(self, dim: int, mlp_dim: int):
        super().__init__()
        self.norm = nn.RMSNorm(dim, eps=1e-6)
        self.down = nn.Linear(dim, mlp_dim)
        self.gate = nn.Linear(dim, mlp_dim)
        self.up = nn.Linear(mlp_dim, dim)
        self.act = nn.SiLU()
        self.residual_logit = nn.Parameter(torch.tensor(-2.0))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        h = self.norm(x)
        h = self.up(self.act(self.down(h)) * torch.sigmoid(self.gate(h)))
        return x + torch.sigmoid(self.residual_logit) * h


class GatedResidualFusion(nn.Module):
    """
    门控残差旁路(新架构):
      branch(h) = gate_out · adapter2( small_mid( adapter1(h) ) )
    其中 small_mid = 0.6B 的 1/3~2/3 层(冻结,仅作可导计算通道)。
    初始: gate_out = sigmoid(-10) ≈ 0 且 adapter2 零初始化 → branch ≡ 0,
    严格保持原 4B 输出不变(identity preservation)。
    """

    def __init__(self, small_model, s1: int, s2: int,
                 d_large: int, d_small: int, mlp_dim: int, bridge_depth: int = 1):
        super().__init__()
        # 输入/输出归一化(Qwen3 同款 RMSNorm):
        #   输入侧: h12(范数~150)先归一化到单位尺度,适配器不受残差流绝对量级影响
        #   输出侧: branch 归一化到 √d_large(≈50)量级,与 4B 残差流(范数~150)同量级再加回
        self.input_norm = nn.RMSNorm(d_large, eps=1e-6)
        self.output_norm = nn.RMSNorm(d_large, eps=1e-6)
        self.adapter1 = GatedAdapter(d_large, d_small, mlp_dim, bridge_depth)
        self.adapter2 = GatedAdapter(d_small, d_large, mlp_dim, bridge_depth)
        # 0.6B 的 1/3~2/3 层(冻结使用: 不更新参数,但梯度可穿过它回传)。
        # 用普通 list 而非 ModuleList: 这些层属于小模型本体,不应混入 fusion 的 state_dict
        self.small_layers = [small_model.model.layers[i] for i in range(s1, s2 + 1)]
        for layer in self.small_layers:
            for p in layer.parameters():
                p.requires_grad_(False)
        # 旁路级门控: 初始 -10 → sigmoid ≈ 4.5e-5 ≈ 0(初始旁路不生效)
        self.gate_logit = nn.Parameter(torch.tensor(-10.0))
        self.small_dev = None
        self.small_dtype = None

    def forward(self, h):
        """h: (B, L, d_large) 为 4B 在 1/3 处(第 pos1 层输出)的隐状态;
        返回 (B, L, d_large) 的旁路贡献,由钩子加回 2/3 处残差流"""
        x = self.input_norm(h)                           # ① 输入归一化 → 单位尺度
        x = self.adapter1(x)
        x = x.to(device=self.small_dev, dtype=self.small_dtype)
        # 零拷贝复刻 Qwen3Model.forward 的层循环(用真实层对象 + 真实 rotary + 官方掩码):
        # 与正常前向逐参一致,绕开一切"子模型重建/深拷贝"的组件状态不一致问题
        from transformers.models.qwen3.modeling_qwen3 import (
            create_causal_mask, create_sliding_window_causal_mask)
        sm = self.small_model_ref
        B, L, _ = x.shape
        pos_ids = torch.arange(L, device=self.small_dev).unsqueeze(0)
        cache_position = torch.arange(L, device=self.small_dev)   # transformers 5.3 需要该参数
        pos_emb = sm.rotary_emb(x, pos_ids)                      # 真实 rotary: cos/sin (B, L, D)
        # 与官方 Qwen3Model.forward 一致: create_causal_mask 返回的是「单个掩码」(张量或 None),
        # 不是字典。需要自己按层类型组装映射表;None 直接传给层的 attention_mask 即可
        # (层内部会自行构建因果掩码)。
        mask_kwargs = dict(config=sm.config, inputs_embeds=x,
                           attention_mask=None, cache_position=cache_position,
                           past_key_values=None, position_ids=pos_ids)
        mask_map = {"full_attention": create_causal_mask(**mask_kwargs)}
        if getattr(sm, "has_sliding_layers", False):
            mask_map["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
        h = x
        # layer_types 兜底(与官方 Qwen3Config.__post_init__ 等价): 0.6B 无滑动窗口 → 全层 full_attention
        layer_types = getattr(sm.config, "layer_types", None)
        if layer_types is None:
            layer_types = ["full_attention"] * sm.config.num_hidden_layers
        if not self.bypass_small:
            for j, layer in enumerate(sm.layers[self.s1:self.s2 + 1]):
                mask = mask_map[layer_types[self.s1 + j]]
                # transformers 5.x 的层直接返回张量，不能加 [0]。
                h = layer(h, attention_mask=mask,
                          position_embeddings=pos_emb, position_ids=pos_ids,
                          past_key_values=None, use_cache=False)
        x = sm.norm(h)
        x = x.to(device=h.device, dtype=h.dtype)
        out = self.adapter2(x)
        out = self.output_norm(out)                      # ② 输出归一化 → 与残差流同量级
        return out * torch.sigmoid(self.gate_logit)


def resolve_small_range(config: Config, n_small: int):
    """解析 0.6B 旁路层范围(含端点, 0-based, 已夹紧合规)。"""
    s1 = config.fusion_small_start if config.fusion_small_start is not None \
        else int(config.fusion_pos1_frac * n_small)
    s2 = config.fusion_small_end if config.fusion_small_end is not None \
        else int(config.fusion_pos2_frac * n_small)
    if s1 < 0:      # 负索引从末尾数(-1 = 最后一层)
        s1 += n_small
    if s2 < 0:
        s2 += n_small
    # 参数合规: 夹到 [0, n_small-1] 且保证 s1 ≤ s2
    s1 = max(0, min(s1, n_small - 1))
    s2 = max(s1, min(s2, n_small - 1))
    return s1, s2


def resolve_large_range(config: Config, n_large: int):
    """解析 4B 接入位置(取隐状态层 l1, 加回残差层 l2; 0-based, 已夹紧, 保证 l2 > l1)。"""
    l1 = config.fusion_large_start if config.fusion_large_start is not None \
        else int(config.fusion_pos1_frac * n_large)
    l2 = config.fusion_large_end if config.fusion_large_end is not None \
        else int(config.fusion_pos2_frac * n_large)
    l1 = max(0, min(l1, n_large - 2))
    l2 = max(l1 + 1, min(l2, n_large - 1))
    return l1, l2


def attach_fusion(model_large, model_small, config: Config) -> GatedResidualFusion:
    """
    用前向钩子把门控残差旁路挂到 4B 上:
      pos1 = 36×1/3 = 第 12 层输出 → 捕获隐状态(不改输出)
      pos2 = 36×2/3 = 第 24 层输出 → 加上 gated branch
    返回 fusion 模块(其参数可训练,当前仅搭建架构)。
    """
    n_large = model_large.config.num_hidden_layers
    n_small = model_small.config.num_hidden_layers
    l1, l2 = resolve_large_range(config, n_large)
    s1, s2 = resolve_small_range(config, n_small)
    fusion = GatedResidualFusion(
        model_small, s1, s2,
        d_large=model_large.config.hidden_size,
        d_small=model_small.config.hidden_size,
        mlp_dim=config.fusion_mlp_dim,
        bridge_depth=config.fusion_bridge_depth,
    )
    # 注意: 只搬适配器与门控到 4B 的设备+精度(bf16);不能对整个 fusion 调 .to() ——
    # small_layers 是 0.6B 模型本身的层对象,整体搬会挪走/毁掉小模型的参数位置
    dev_l = model_device(model_large)
    dt_large = next(model_large.parameters()).dtype
    fusion.input_norm.to(device=dev_l, dtype=dt_large)
    fusion.output_norm.to(device=dev_l, dtype=dt_large)
    fusion.adapter1.to(device=dev_l, dtype=dt_large)
    fusion.adapter2.to(device=dev_l, dtype=dt_large)
    fusion.gate_logit.data = fusion.gate_logit.data.to(device=dev_l)  # 标量保持 float32
    fusion.small_dev = model_device(model_small)
    fusion.small_dtype = next(model_small.parameters()).dtype
    # 零拷贝: 只存真实 0.6B 模型本体的引用与层切片下标,forward 里手工复刻官方层循环。
    # 用 object.__setattr__ 挂载,不注册为 fusion 的子模块,避免 state_dict 混入冻结权重
    object.__setattr__(fusion, "small_model_ref", model_small.model)
    object.__setattr__(fusion, "s1", s1)
    object.__setattr__(fusion, "s2", s2)
    object.__setattr__(fusion, "bypass_small", bool(config.fusion_bypass_small))

    state = {}
    # 训练诊断用(训练脚本通过这几个属性控制/读取钩子):
    #   enabled=False     → 旁路关闭, 用于算 baseline loss
    #   branch_override   → 不为 None 时, 注入这个 h 而不是捕获到的 h(对比损失用)
    #   hook_state        → 暴露捕获到的 h12 给训练脚本
    object.__setattr__(fusion, "enabled", True)
    object.__setattr__(fusion, "branch_override", None)
    object.__setattr__(fusion, "hook_state", state)

    def capture(module, args, output):
        # transformers 5.x: 层输出是张量 (B, L, d); 旧版可能是 (hidden_states, ...) 元组
        state["h"] = output[0] if isinstance(output, tuple) else output
        return output

    def inject(module, args, output):
        if not fusion.enabled:
            return output
        h_src = state.get("h") if fusion.branch_override is None else fusion.branch_override
        if h_src is None:
            return output
        branch = fusion(h_src)
        if isinstance(output, tuple):
            h = output[0]
            return (h + branch.to(h.dtype),) + output[1:]
        return output + branch.to(output.dtype)

    model_large.model.layers[l1].register_forward_hook(capture)
    model_large.model.layers[l2].register_forward_hook(inject)
    n_branch = sum(p.numel() for p in fusion.adapter1.parameters()) \
               + sum(p.numel() for p in fusion.adapter2.parameters())
    print(f"    门控残差旁路已挂载: 4B 第{l1}层输出取隐状态 → 0.6B 第{s1}~{s2}层 "
          f"→ 4B 第{l2}层加回 | 适配器可训练参数 {n_branch/1e6:.1f}M")
    return fusion


def save_fusion(fusion: GatedResidualFusion, path: str):
    """保存旁路参数 + 架构元数据(层范围/维度), 便于加载时校验参数合规性。"""
    payload = {
        "state_dict": fusion.state_dict(),
        "meta": {
            "small_s1": int(getattr(fusion, "s1", -1)),
            "small_s2": int(getattr(fusion, "s2", -1)),
            "mlp_dim": int(fusion.adapter1.down.out_features),
            "bridge_depth": 1 + len(fusion.adapter1.blocks),
            "d_large": int(fusion.adapter1.down.in_features),
            "d_small": int(fusion.adapter1.up.out_features),
            "bypass_small": bool(getattr(fusion, "bypass_small", False)),
        },
    }
    torch.save(payload, path)
    print(f"    旁路参数已保存: {path}")


def load_fusion(fusion: GatedResidualFusion, path: str):
    """加载旁路参数; 兼容旧版裸 state_dict, 并对新格式做层范围/维度校验。"""
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        sd = obj["state_dict"]
        meta = obj.get("meta", {})
        if meta:
            cur = (int(getattr(fusion, "s1", -1)), int(getattr(fusion, "s2", -1)),
                   int(fusion.adapter1.down.out_features),
                   1 + len(fusion.adapter1.blocks))
            new = (meta.get("small_s1"), meta.get("small_s2"), meta.get("mlp_dim"),
                   meta.get("bridge_depth", 1))
            if new[0] is not None and cur != new:
                print(f"    [警告] 检查点层范围/维度与当前不符: "
                      f"检查点 s1={new[0]} s2={new[1]} mlp={new[2]} depth={new[3]} "
                      f"vs 当前 s1={cur[0]} s2={cur[1]} mlp={cur[2]} depth={cur[3]}")
            saved_bypass = meta.get("bypass_small")
            current_bypass = bool(getattr(fusion, "bypass_small", False))
            if saved_bypass is not None and bool(saved_bypass) != current_bypass:
                raise ValueError("检查点 bypass_small 与当前配置不一致；"
                                 "bridge-only 权重评测时请传 --bypass_small")
    else:
        sd = obj   # 旧版裸 state_dict
    fusion.load_state_dict(sd)
    print(f"    旁路参数已加载: {path}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  ⑧ 完整融合管线(单次推理封装)                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def fused_inference(model_large, tokenizer, config: Config, prompt: str):
    """
    门控残差融合推理: 4B 正常前向(第 pos1→pos2 层之间挂门控旁路)+ 文本解码。
    调用前必须先 attach_fusion 把旁路钩子挂到 4B 上。
    返回 (answer, stats)。
    """
    t0 = time.time()
    text = build_prompt_text(tokenizer, prompt)
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    dev_l = model_device(model_large)
    embeds_l = model_large.model.embed_tokens(input_ids.to(dev_l))
    attention_mask = torch.ones((1, input_ids.shape[1]), device=dev_l)
    # 预填充: 跑一遍 4B 得到最后隐状态与 KV cache(钩子在此过程中注入门控旁路)
    with torch.inference_mode():
        outputs = model_large.model(inputs_embeds=embeds_l,
                                    attention_mask=attention_mask, use_cache=True)
    past = outputs.past_key_values
    h_last = outputs.last_hidden_state[:, -1:, :]
    answer = decode_answer(model_large, tokenizer, h_last, past, config)

    stats = {
        "prompt_tokens": int(input_ids.shape[1]),
        "max_new_tokens": config.max_new_tokens,
        "time_s": round(time.time() - t0, 2),
    }
    return answer, stats


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  ⑨ 评测功能: 跑测试题集并保存到文件                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def run_evaluation(model_large, tokenizer, config: Config) -> str:
    """
    评测模式(Config.enable_eval=True 时启用):
    对 eval_questions.py 的测试题集逐题跑融合管线,结果保存到 eval_results/fused_<时间戳>.json。
    与 test_baseline.py 的基线结果对比即可看出融合旁路的增益。返回保存路径。
    """
    os.makedirs("eval_results", exist_ok=True)
    results = []
    for i, q in enumerate(EVAL_QUESTIONS, 1):
        print(f"\n  ── 第 {i}/{len(EVAL_QUESTIONS)} 题: {q['question'][:40]} ...")
        try:
            answer, stats = fused_inference(model_large, tokenizer, config, q["question"])
            stats.update({"id": q["id"], "question": q["question"],
                          "expected": q["answer"], "model_answer": answer.strip()})
            print(f"    回答: {stats['model_answer'][:80]}")
        except Exception as e:  # 单题失败不中断整批
            stats = {"id": q["id"], "question": q["question"],
                     "expected": q["answer"], "error": str(e)}
            print(f"    [失败] {e}")
        results.append(stats)

    path = os.path.join("eval_results", f"fused_{time.strftime('%Y%m%d_%H%M%S')}.json")
    payload = {
        "meta": {
            "script": "main(fused)",
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "torch": torch.__version__,
            "cuda": torch.cuda.is_available(),
            "models": {"small": config.model_small_id, "large": config.model_large_id},
            "params": {
                "fusion_pos1_frac": config.fusion_pos1_frac,
                "fusion_pos2_frac": config.fusion_pos2_frac,
                "fusion_mlp_dim": config.fusion_mlp_dim,
                "max_new_tokens": config.max_new_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "seed": config.seed},
            "question_count": len(EVAL_QUESTIONS),
        },
        "results": results,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  ⑩ 主流程                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def main():
    config = Config()
    torch.manual_seed(config.seed)  # 固定随机种子,保证解码采样可复现
    print("=" * 62)
    print("门控残差融合(4B + 0.6B 中段旁路)")
    print(f"环境: torch {torch.__version__} | CUDA "
          f"{'可用' if torch.cuda.is_available() else '不可用(纯 CPU,速度较慢)'}")
    print("=" * 62)

    # [1/3] 加载模型与分词器(0.6B 作为旁路中段层的宿主)
    print("\n[1/3] 加载模型与分词器 ...")
    dt = resolve_dtype(config.dtype)
    need_small = config.fusion_enabled
    large_path = resolve_model_path(config.model_large_id, config.model_large_local)
    model_large = AutoModelForCausalLM.from_pretrained(
        large_path, dtype=dt, device_map=config.device_map)
    model_large.eval()
    model_small = None
    if need_small:
        small_path = resolve_model_path(config.model_small_id, config.model_small_local)
        model_small = AutoModelForCausalLM.from_pretrained(
            small_path, dtype=dt, device_map=config.device_map_small or config.device_map)
        model_small.eval()
        tokenizer = AutoTokenizer.from_pretrained(small_path)   # 两模型共用同一分词器
        print(f"    0.6B: {model_small.config.num_hidden_layers} 层, 隐维 {model_small.config.hidden_size}")
    else:
        tokenizer = AutoTokenizer.from_pretrained(large_path)
    print(f"    4B : {model_large.config.num_hidden_layers} 层, 隐维 {model_large.config.hidden_size}")

    # [2/3] 挂载门控残差旁路(初始 gate≈0 + 零初始化投影 → 严格保持原 4B 行为)
    if config.fusion_enabled:
        attach_fusion(model_large, model_small, config)

    # [3/3] 融合推理: 评测模式(跑测试题集并保存) 或 单条演示
    if config.enable_eval:
        print(f"\n[3/3] 评测模式: 对 {len(EVAL_QUESTIONS)} 道测试题逐题跑融合管线 ...")
        path = run_evaluation(model_large, tokenizer, config)
        print("\n" + "=" * 62)
        print(f"评测完成, 结果已保存: {path}")
        print("=" * 62)
    else:
        print("\n[3/3] 融合推理 ...")
        answer, stats = fused_inference(model_large, tokenizer, config, config.prompt)
        print("\n" + "=" * 62)
        print("最终回答:")
        print(answer)
        print("-" * 62)
        print(f"统计: 输入 {stats['prompt_tokens']} tokens | 耗时 {stats['time_s']}s")
        print("=" * 62)


if __name__ == "__main__":
    main()
