#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2d感觉实验/train_world_num.py — 数值世界模型（纯数值输入，MLP，归一化）

输入：obs（56 维：角度/位置/目标/距离/方向/感觉） + 动作（2 维，/15）
输出：预测 delta（位置变化 x,y + 触觉 + 痛 + 负载）
训练：监督 MSE
保存：world_num.pt

用法：
  python train_world_num.py --device cuda --steps 5000
"""

import argparse
import random

import torch
import torch.nn.functional as F

import env


class WorldNum(torch.nn.Module):
    def __init__(self, in_dim=58, hidden=256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, 5),
        )

    def forward(self, obs, action):
        return self.net(torch.cat([obs, action], dim=-1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--save", default="world_num.pt")
    args = p.parse_args()

    device = torch.device(args.device)
    model = WorldNum().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = random.Random(0)
    run = 0.0
    for step in range(args.steps):
        t1 = rng.uniform(-180, 180)
        t2 = rng.uniform(0, 180)
        target = (rng.uniform(-1.8, 1.8), rng.uniform(-1.8, 1.8))
        if env.inverse_kinematics(*target) is None:
            continue
        d1 = rng.uniform(-60, 60)
        d2 = rng.uniform(-60, 60)
        nt1 = t1 + d1
        nt2 = min(env.ELBOW_MAX, max(env.ELBOW_MIN, t2 + d2))
        obs = torch.tensor(env.obs_vector(t1, t2, target), dtype=torch.float32,
                           device=device).unsqueeze(0)
        act = torch.tensor([[d1 / 60.0, d2 / 60.0]], dtype=torch.float32, device=device)
        pred = model(obs, act)
        hx, hy = env.forward_kinematics(t1, t2)
        nhx, nhy = env.forward_kinematics(nt1, nt2)
        actual = torch.tensor([nhx - hx, nhy - hy, 0.0, env.pain(nt1, nt2) / 30.0,
                               env.load(nt1, nt2, None)], device=device)
        loss = F.mse_loss(pred.squeeze(0), actual)
        opt.zero_grad()
        loss.backward()
        opt.step()
        run += loss.item()
        if (step + 1) % 500 == 0:
            print(f"step {step+1} | world mse {run/500:.5f}", flush=True)
            run = 0.0

    torch.save({"model": model.state_dict()}, args.save)
    print(f"[save] {args.save}", flush=True)


def sensitivity_check():
    import env
    t1, t2, target = 30.0, 90.0, (1.0, 1.0)
    obs = torch.tensor(env.obs_vector(t1, t2, target), dtype=torch.float32).unsqueeze(0)
    hx, hy = env.forward_kinematics(t1, t2)
    print("当前手位置:", (round(hx, 2), round(hy, 2)))
    for d1 in (-15, 0, 15):
        out = []
        for d2 in (-15, 0, 15):
            act = torch.tensor([[d1 / 60.0, d2 / 60.0]], dtype=torch.float32)
            with torch.no_grad():
                pred = model(obs, act)
            out.append((round(hx + float(pred[0, 0]), 2), round(hy + float(pred[0, 1]), 2)))
        print(f"动作 ({d1:>3}, ±15) → 预测点位 {out}")


if __name__ == "__main__":
    main()
