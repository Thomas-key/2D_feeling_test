#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pretrain_lora.py — 预训练（睡觉/从零）：LoRA 模型的 base 全量训练

决策：规则世界想象（价值梯度）
世界：规则数据监督（动作 → 状态）
输出：pretrain_base.pt（供 train_day 加载）

用法：
  python pretrain_lora.py --device cuda --steps 10000
"""
import argparse
import random

import torch

import dream
import env
from lora import DecisionLoRA, WorldLoRA
from loop import obs_tensor

def pretrain(dec, world, device, steps, rng):
    """预训练 base：决策（规则世界想象）+ 世界（规则数据监督）。"""
    base_params = list(dec.base.parameters()) + list(dec.head_shoulder.parameters()) + list(dec.head_elbow.parameters())
    wbase_params = list(world.base.parameters()) + list(world.heads.parameters())
    # base 在预训练阶段需要可训练（临时解开）
    for p in base_params + wbase_params:
        p.requires_grad = True
    opt_d = torch.optim.Adam(base_params, lr=3e-4)
    opt_w = torch.optim.Adam(wbase_params, lr=3e-4)
    scale = torch.tensor(env.OUT_SCALE, device=device)
    for step in range(steps):
        smp = dream.sample(rng)          # 数据分布见 dream.py（夜间复习共用）
        if smp is None:
            continue
        t1, t2, target, d1r, d2r = smp
        obs = obs_tensor(t1, t2, target, device)
        # 决策：规则世界想象
        loss_d, v = dream.decision_loss(dec, obs, t1, t2, target)
        opt_d.zero_grad()
        loss_d.backward()
        opt_d.step()
        # 世界：规则数据监督
        act = torch.tensor([[d1r / 60.0, d2r / 60.0]], device=device)
        loss_w = dream.world_loss(world, obs, act, t1, t2, d1r, d2r, scale)
        opt_w.zero_grad()
        loss_w.backward()
        opt_w.step()
        if (step + 1) % 1000 == 0:
            print(f"[pretrain] step {step+1} | 决策价值 {float(v):.3f} | 世界 mse {loss_w.item():.5f}", flush=True)
    # 预训练后冻结 base/heads（只留 LoRA）
    for p in base_params + wbase_params:
        p.requires_grad = False

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    device = torch.device(args.device)
    rng = random.Random(args.seed)
    dec = DecisionLoRA().to(device)
    world = WorldLoRA().to(device)
    pretrain(dec, world, device, args.steps, rng)
    torch.save({"dec_base": dec.base.state_dict(),
                "dec_heads": {k: getattr(dec, k).state_dict()
                              for k in ("head_shoulder", "head_elbow")},
                "world_base": world.base.state_dict(),
                "world_heads": world.heads.state_dict()}, "pretrain_base.pt")
    print("[save] pretrain_base.pt", flush=True)


if __name__ == "__main__":
    main()
