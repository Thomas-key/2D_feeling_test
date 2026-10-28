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
import math
import random

import torch
import torch.nn.functional as F

import env
from lora import DecisionLoRA, WorldLoRA
from train_day import obs_tensor

def pretrain(dec, world, device, steps, rng):
    """预训练 base：决策（规则世界想象）+ 世界（规则数据监督）。"""
    base_params = list(dec.base.parameters()) + list(dec.head_shoulder.parameters()) + list(dec.head_elbow.parameters())
    wbase_params = list(world.base.parameters()) + list(world.heads.parameters())
    opt_d = torch.optim.Adam([p for p in base_params if p.requires_grad] if any(p.requires_grad for p in base_params) else [], lr=3e-4) if False else None
    # base 在预训练阶段需要可训练（临时解开）
    for p in base_params + wbase_params:
        p.requires_grad = True
    opt_d = torch.optim.Adam(base_params, lr=3e-4)
    opt_w = torch.optim.Adam(wbase_params, lr=3e-4)
    for step in range(steps):
        t1 = rng.uniform(-180, 180)
        t2 = rng.uniform(0, 180)
        # 20% 到达 + 10% 接近（目标 = 手 + 小偏移——学小动作微调）
        r0 = rng.random()
        if r0 < 0.2:
            hx0, hy0 = env.forward_kinematics(t1, t2)
            target = (hx0, hy0)
        elif r0 < 0.3:
            hx0, hy0 = env.forward_kinematics(t1, t2)
            ang = rng.uniform(0, 6.28)
            r = rng.uniform(0.05, 0.2)
            target = (hx0 + r * math.cos(ang), hy0 + r * math.sin(ang))
        else:
            target = (rng.uniform(-1.8, 1.8), rng.uniform(-1.8, 1.8))
        if env.inverse_kinematics(*target) is None:
            continue
        obs = obs_tensor(t1, t2, target, device)
        # 决策：规则世界想象
        d1, d2 = dec(obs)
        a1 = (t1 + d1) * math.pi / 180.0
        a2 = (t1 + d1 + (t2 + d2)) * math.pi / 180.0
        hx = torch.cos(a1) + torch.cos(a2)
        hy = torch.sin(a1) + torch.sin(a2)
        dist = torch.sqrt((hx - target[0]) ** 2 + (hy - target[1]) ** 2 + 1e-6)
        nt2 = torch.clamp(t2 + d2, env.ELBOW_MIN, env.ELBOW_MAX)
        pain = torch.relu(nt2 - env.FOLD_PAIN) * 0.5
        v = -dist - 5.0 * (torch.exp(pain / 10.0) - 1.0)
        loss_d = -v
        opt_d.zero_grad()
        loss_d.backward()
        opt_d.step()
        # 世界：规则数据监督（30% 精细动作——小动作学准）
        if rng.random() < 0.3:
            d1r = rng.uniform(-10, 10)
            d2r = rng.uniform(-10, 10)
        else:
            d1r = rng.uniform(-60, 60)
            d2r = rng.uniform(-60, 60)
        act = torch.tensor([[d1r / 60.0, d2r / 60.0]], device=device)
        pred = world(obs, act)
        nt1 = t1 + d1r
        nnt2 = min(env.ELBOW_MAX, max(env.ELBOW_MIN, t2 + d2r))
        nhx, nhy = env.forward_kinematics(nt1, nnt2)
        ohx, ohy = env.forward_kinematics(t1, t2)
        # 输出归一化（去量纲——每头各自尺度）
        actual_raw = torch.tensor([nhx - ohx, nhy - ohy, 0.0, env.pain(nt1, nnt2),
                                   env.load(nt1, nnt2, None)], device=device)
        scale = torch.tensor(env.OUT_SCALE, device=device)
        actual = actual_raw / scale
        # 位置项相对误差权重（小动作放大——重视小动作）；其他项线性
        diff = pred.squeeze(0) - actual
        w_pos = 1.0 / (actual[:2].abs() + 0.05) ** 2     # 小动作权重大
        loss_pos = (diff[:2] ** 2 * w_pos).mean()
        loss_other = (diff[2:] ** 2).mean()
        loss_w = loss_pos + loss_other
        # 痛项：指数差（pred/actual 的痛——归一化值 ×30 还原痛度）
        pain_pred = torch.exp(torch.clamp(pred[0, 3] * 30.0, -30.0, 60.0) / 10.0)
        pain_act = torch.exp(torch.clamp(actual[3] * 30.0, -30.0, 60.0) / 10.0)
        loss_w = loss_w + 0.5 * (pain_pred - pain_act) ** 2
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
