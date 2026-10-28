#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_enum.py — 状态枚举动作：世界预测价值 vs 实际价值（找世界误判）"""
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
p.add_argument("--target", nargs=2, type=float, default=[-1.0, -1.0])
p.add_argument("--t1", type=float, default=-175.1)
p.add_argument("--t2", type=float, default=90.3)
args = p.parse_args()
device = torch.device(args.device)
world = WorldLoRA().to(device)
ckpt = torch.load(args.base, map_location="cpu")
world.base.load_state_dict(ckpt["world_base"])
world.heads.load_state_dict(ckpt["world_heads"])
world.eval()

target = (args.target[0], args.target[1])
t1, t2 = args.t1, args.t2
hx, hy = env.forward_kinematics(t1, t2)


def true_value(a1, a2):
    ax, ay = env.forward_kinematics(a1, a2)
    dist = math.hypot(ax - target[0], ay - target[1])
    p = env.pain(a1, a2)
    return -dist - 5.0 * (math.exp(min(p, 30.0) / 10.0) - 1.0) - 2.0 * env.load(a1, a2, None)


v_cur = true_value(t1, t2)
print(f"状态 ({t1},{t2}) 手=({hx:.2f},{hy:.2f}) 目标={target} 当前价值 {v_cur:.3f}")
print(f"{'动作':>10} | {'世界预测价值':>10} | {'实际价值':>8} | {'世界误判?':>8}")
for d1 in (-10, -5, -2, 0, 2, 5, 10):
    for d2 in (-5, 0, 5):
        obs = torch.tensor(env.obs_vector(t1, t2, target), dtype=torch.float32, device=device).unsqueeze(0)
        act = torch.tensor([[d1 / 60.0, d2 / 60.0]], dtype=torch.float32, device=device)
        with torch.no_grad():
            pred = world(obs, act) * torch.tensor(env.OUT_SCALE, device=device)
        nhx, nhy = hx + float(pred[0, 0]), hy + float(pred[0, 1])
        pain = max(float(pred[0, 3]), 0.0)   # 物理上不可能为负（与 world_value 一致）
        vw = -math.hypot(nhx - target[0], nhy - target[1]) \
             - 5.0 * (math.exp(min(pain, 30.0) / 10.0) - 1.0) - 2.0 * max(float(pred[0, 4]), 0.0)
        nt1, nt2, want2, pain_a, load_a = env.step_body(t1, t2, d1, d2)
        ax, ay = env.forward_kinematics(nt1, nt2)
        va = (-math.hypot(ax - target[0], ay - target[1])
              - 5.0 * (math.exp(min(pain_a, env.PAIN_CAP) / 10.0) - 1.0)
              - 2.0 * load_a)
        mis = "误判!" if (vw <= v_cur + 0.01 and va > v_cur + 0.01) else ""
        print(f"({d1:>3},{d2:>3}) | {vw:>10.3f} | {va:>8.3f} | {mis:>8}")
