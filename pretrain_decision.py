#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2d感觉实验/pretrain_decision.py — 预训练决策（想象梯度，规则世界可微）

规则世界（可微，torch 实现）：
  新角度 = 当前 + 动作（clamp 肘 [0,180]）
  手位置 = 正运动学
  痛 = 超出 [0,180] + 折叠（>160）
  负载 = 折叠/球
  价值 = -距离 - 5·(exp(痛/10)-1) - 2·负载
训练：决策 → 动作 → 规则世界（可微）→ 价值 → loss = -V → 梯度（只更新决策）

用法：
  python pretrain_decision.py --device cuda --steps 5000
"""

import argparse
import math
import random

import torch
import torch.nn.functional as F

import env


def obs_tensor(t1, t2, target, device):
    return torch.tensor(env.obs_vector(t1, t2, target), dtype=torch.float32,
                        device=device).unsqueeze(0)


def rule_value(t1, t2, target, d1, d2):
    """规则世界（可微）：动作 → 新状态 → 价值。"""
    t1r = t1 * math.pi / 180.0
    nt1 = t1 + d1
    nt2 = torch.clamp(t2 + d2, env.ELBOW_MIN, env.ELBOW_MAX)
    a1 = nt1 * math.pi / 180.0
    a2 = (nt1 + nt2) * math.pi / 180.0
    hx = torch.cos(a1) + torch.cos(a2)
    hy = torch.sin(a1) + torch.sin(a2)
    dist = torch.sqrt((hx - target[0]) ** 2 + (hy - target[1]) ** 2 + 1e-6)
    pain = torch.relu(nt2 - env.FOLD_PAIN) * 0.5
    load = torch.clamp((nt2 - env.FOLD_PAIN) / 20.0, 0.0, 1.0)
    return -dist - 5.0 * (torch.exp(pain / 10.0) - 1.0) - 2.0 * load


class DecisionNum(torch.nn.Module):
    def __init__(self, in_dim=56, hidden=256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, 2),
        )

    def forward(self, obs):
        return torch.tanh(self.net(obs)) * 60.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--save", default="decision_pre.pt")
    args = p.parse_args()

    device = torch.device(args.device)
    model = DecisionNum().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = random.Random(0)
    run = 0.0
    for step in range(args.steps):
        t1 = rng.uniform(-180, 180)
        t2 = rng.uniform(0, 180)
        target = (rng.uniform(-1.8, 1.8), rng.uniform(-1.8, 1.8))
        if env.inverse_kinematics(*target) is None:
            continue
        obs = obs_tensor(t1, t2, target, device)
        action = model(obs)
        v = rule_value(t1, t2, target, action[0, 0], action[0, 1])
        loss = -v
        opt.zero_grad()
        loss.backward()
        opt.step()
        run += float(v)
        if (step + 1) % 500 == 0:
            print(f"step {step+1} | 平均价值 {run/500:.3f}", flush=True)
            run = 0.0

    torch.save({"model": model.state_dict()}, args.save)
    print(f"[save] {args.save}", flush=True)


if __name__ == "__main__":
    main()
