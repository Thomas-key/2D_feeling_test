#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_value.py — 价值分解（距离/痛/负载）在指定状态下"""
import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import env  # noqa: E402
from train_world_num import WorldNum


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--world", default="world_num.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--t1", type=float, default=129.0)
    p.add_argument("--t2", type=float, default=115.0)
    p.add_argument("--target", nargs=2, type=float, default=[-0.7, -0.2])
    args = p.parse_args()
    device = torch.device(args.device)
    world = WorldNum().to(device)
    world.load_state_dict(torch.load(args.world, map_location="cpu")["model"])
    world.eval()

    t1, t2 = args.t1, args.t2
    target = (args.target[0], args.target[1])
    hx, hy = env.forward_kinematics(t1, t2)
    print(f"状态 ({t1:.0f},{t2:.0f}) 手=({hx:.2f},{hy:.2f}) 目标={target} 距离={math.hypot(hx-target[0],hy-target[1]):.2f}")
    obs = torch.tensor(env.obs_vector(t1, t2, target), dtype=torch.float32, device=device).unsqueeze(0)
    print(f"{'动作':>12} | {'预测距离':>8} | {'痛':>6} | {'负载':>6} | {'价值':>7}")
    for d1 in (-30, -15, 0, 15, 30):
        for d2 in (-30, -15, 0, 15, 30):
            act = torch.tensor([[d1 / 60.0, d2 / 60.0]], device=device)
            with torch.no_grad():
                pred = world(obs, act)
            nx, ny = hx + float(pred[0, 0]), hy + float(pred[0, 1])
            dist = math.hypot(nx - target[0], ny - target[1])
            pain = float(pred[0, 3]) * 30.0
            load = float(pred[0, 4])
            v = -dist - 5.0 * (math.exp(min(pain, 30.0) / 10.0) - 1.0) - 2.0 * load
            if d1 in (-30, 0, 30) and d2 in (-30, 0, 30):
                print(f"({d1:>3},{d2:>3}) | {dist:8.2f} | {pain:6.1f} | {load:6.2f} | {v:7.2f}")


if __name__ == "__main__":
    main()
