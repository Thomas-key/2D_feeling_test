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
from collections import deque

import torch

import dream
import env
from lora import DecisionLoRA, WorldLoRA
from loop import obs_tensor


def build_val(n=500):
    """固定验证集：seed 999（与训练 seed 0 不重叠），内容不参与梯度。"""
    rng = random.Random(999)
    val = []
    while len(val) < n:
        smp = dream.sample(rng)
        if smp is not None:
            val.append(smp)
    return val


def evaluate(dec, world, val, device, scale):
    """验证集上的平均损失：(决策, 位置, 痛)。"""
    with torch.no_grad():
        ld = lp = ln = 0.0
        for t1, t2, target, d1r, d2r in val:
            obs = obs_tensor(t1, t2, target, device)
            act = torch.tensor([[d1r / 60.0, d2r / 60.0]], device=device)
            _, pos, pain = dream.world_losses(world, obs, act, t1, t2, d1r, d2r, scale)
            loss_d, _v = dream.decision_loss(dec, obs, t1, t2, target)
            ld += float(loss_d)
            lp += float(pos)
            ln += float(pain)
        n = len(val)
        return ld / n, lp / n, ln / n


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
    val = build_val()
    recent = deque(maxlen=200)       # 最近 200 步的滑动平均（单样本损失噪声太大）
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
        loss_w, loss_pos, loss_pain = dream.world_losses(world, obs, act, t1, t2, d1r, d2r, scale)
        opt_w.zero_grad()
        loss_w.backward()
        opt_w.step()
        recent.append((float(loss_d), float(loss_pos), float(loss_pain)))
        if (step + 1) % 1000 == 0:
            n = len(recent)
            td = sum(x[0] for x in recent) / n
            tp = sum(x[1] for x in recent) / n
            tn = sum(x[2] for x in recent) / n
            vd, vp, vn = evaluate(dec, world, val, device, scale)
            print(f"[pretrain] {step+1} | train 决策 {td:.3f} 位置 {tp:.4f} 痛 {tn:.4f}"
                  f" | val 决策 {vd:.3f} 位置 {vp:.4f} 痛 {vn:.4f}", flush=True)
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
