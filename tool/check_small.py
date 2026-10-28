#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_small.py — 世界模型：小动作 vs 大动作 预测误差对比"""
import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import env
from lora import WorldLoRA

p = argparse.ArgumentParser()
p.add_argument("--device", default="cuda")
p.add_argument("--base", default="pretrain_base.pt")
args = p.parse_args()
device = torch.device(args.device)
world = WorldLoRA().to(device)
ckpt = torch.load(args.base, map_location="cpu")
world.base.load_state_dict(ckpt["world_base"])
world.heads.load_state_dict(ckpt["world_heads"])
world.eval()

t1, t2, target = 30.0, 90.0, (-1.0, -1.0)
hx, hy = env.forward_kinematics(t1, t2)
print(f"状态 (30,90) 手=({hx:.2f},{hy:.2f}) 目标={target}")
print(f"{'动作':>10} | {'预测点位':>16} | {'实际点位':>16} | {'误差':>6}")
for d1, d2 in [(5, 0), (-5, 0), (10, 0), (-10, 0), (20, 10), (30, 0), (60, 0), (-60, 0)]:
    obs = torch.tensor(env.obs_vector(t1, t2, target), dtype=torch.float32, device=device).unsqueeze(0)
    act = torch.tensor([[d1 / 60.0, d2 / 60.0]], dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = world(obs, act) * torch.tensor(env.OUT_SCALE, device=device)
    px, py = hx + float(pred[0, 0]), hy + float(pred[0, 1])
    ax, ay = env.forward_kinematics(t1 + d1, min(env.FOLD_LIMIT, max(env.ELBOW_MIN, t2 + d2)))
    err = math.hypot(px - ax, py - ay)
    print(f"({d1:>3},{d2:>3}) | ({px:>6.2f},{py:>6.2f}) | ({ax:>6.2f},{ay:>6.2f}) | {err:>5.3f}")
