#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""lora.py — LoRA（低秩适配）+ 拆头模型（决策 2 头 / 世界 5 头）"""

import torch
import torch.nn.functional as F


class LoRALinear(torch.nn.Module):
    """包住 Linear：base 冻结 + A/B 低秩适配（只训练 A/B）。"""

    def __init__(self, base, r=8, alpha=16.0):
        super().__init__()
        self.base = base
        for prm in self.base.parameters():
            prm.requires_grad = False
        in_dim, out_dim = base.weight.shape[1], base.weight.shape[0]
        self.A = torch.nn.Parameter(torch.randn(in_dim, r) * 0.01)
        self.B = torch.nn.Parameter(torch.zeros(out_dim, r))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A @ self.B.T) * self.scale

    @torch.no_grad()
    def merge(self):
        """把 A/B 合并进 base（base.weight += scale·B·A）——睡眠巩固。"""
        self.base.weight.data += (self.B @ self.A.T) * self.scale
        self.A.data.zero_()
        self.B.data.zero_()

    def reset(self):
        self.A.data.zero_()
        self.B.data.zero_()


class DecisionLoRA(torch.nn.Module):
    """决策：base（冻结）+ 输出拆 2 头（大臂/小臂）+ 各 LoRA。"""

    def __init__(self, in_dim=59, hidden=256, r=8):
        super().__init__()
        self.in_dim = in_dim
        self.base = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
        )
        for prm in self.base.parameters():
            prm.requires_grad = False
        # 输出拆头：大臂/小臂
        self.head_shoulder = torch.nn.Linear(hidden, 1)
        self.head_elbow = torch.nn.Linear(hidden, 1)
        for prm in list(self.head_shoulder.parameters()) + list(self.head_elbow.parameters()):
            prm.requires_grad = False
        # LoRA：base 最后一层 + 两个头（简化：在 head 上加 LoRA）
        self.lora_shoulder = LoRALinear(self.head_shoulder, r)
        self.lora_elbow = LoRALinear(self.head_elbow, r)

    def forward(self, obs):
        h = self.base(obs)
        d1 = torch.tanh(self.lora_shoulder(h)) * 60.0
        d2 = torch.tanh(self.lora_elbow(h)) * 60.0
        return d1, d2          # 大臂/小臂动作（度）

    def loras(self):
        return {"shoulder": self.lora_shoulder, "elbow": self.lora_elbow}


class WorldLoRA(torch.nn.Module):
    """世界：base（冻结）+ 输出拆 5 头（位置x/y/触觉/痛/负载）+ 各 LoRA。"""

    def __init__(self, in_dim=59, hidden=256, r=8):
        super().__init__()
        # 动作单独嵌入（放大——避免被 obs 淹没）
        self.act_proj = torch.nn.Sequential(
            torch.nn.Linear(2, 64), torch.nn.ReLU(), torch.nn.Linear(64, hidden))
        self.base = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
        )
        for prm in self.base.parameters():
            prm.requires_grad = False
        self.heads = torch.nn.ModuleDict({
            "px": torch.nn.Linear(hidden, 1), "py": torch.nn.Linear(hidden, 1),
            "touch": torch.nn.Linear(hidden, 1), "pain": torch.nn.Linear(hidden, 1),
            "load": torch.nn.Linear(hidden, 1),
        })
        for prm in self.heads.parameters():
            prm.requires_grad = False
        self.loras = torch.nn.ModuleDict({
            k: LoRALinear(v, r) for k, v in self.heads.items()
        })

    def forward(self, obs, action):
        h = self.base(obs) + 3.0 * self.act_proj(action)   # 动作放大嵌入
        return torch.cat([self.loras[k](h) for k in self.heads], dim=-1)

    def loras_dict(self):
        return self.loras
