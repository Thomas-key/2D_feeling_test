#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_policy.py — 世界模型的"验证"会不会判错？按动作大小分桶统计

回答一个具体问题：世界模型对大动作的位置误差（0.4~0.7 米）到底影不影响决策？
做法：随机采样 (状态, 目标) × 一批候选动作，比较
    世界判定：world_value > true_value(当前) + 0.01
    真实判定：执行后真实价值 > true_value(当前) + 0.01
分桶统计 **误拒**（真能变好但被拒绝）和 **误接**（真变差却被放行），
以及平均 |ΔV 误差|。

用法：python tool/check_policy.py --device cuda --samples 200
"""

import argparse
import math
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import env
from lora import DecisionLoRA, WorldLoRA
from loop import load_base, obs_tensor, true_value, world_value

ACCEPT_EPS = 0.01
ACTIONS = [d for d in (-60, -30, -10, -2, 0, 2, 10, 30, 60)]
BUCKETS = [("小 ≤10°", 0.0, 10.0), ("中 10~30°", 10.0, 30.0), ("大 30~60°", 30.0, 60.0)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--base", default="pretrain_base.pt")
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device(args.device)
    dec = DecisionLoRA().to(device)
    world = WorldLoRA().to(device)
    load_base(dec, world, args.base)
    scale = torch.tensor(env.OUT_SCALE, device=device)
    rng = random.Random(args.seed)

    stat = {k: {"n": 0, "err_rej": 0, "err_acc": 0, "abs_err": 0.0}
            for k, _, _ in BUCKETS}

    for _ in range(args.samples):
        t1 = rng.uniform(-180.0, 180.0)
        t2 = rng.uniform(env.ELBOW_MIN, env.FOLD_MAX)
        target = (rng.uniform(-1.8, 1.8), rng.uniform(-1.8, 1.8))
        hx, hy = env.forward_kinematics(t1, t2)
        v_cur = true_value(t1, t2, target)
        obs = obs_tensor(t1, t2, target, device)
        for d1 in ACTIONS:
            for d2 in ACTIONS:
                m = float(max(abs(d1), abs(d2)))
                key = next((k for k, lo, hi in BUCKETS if lo < m <= hi), None)
                if key is None:
                    continue
                with torch.no_grad():
                    act = torch.tensor([[d1 / 60.0, d2 / 60.0]], device=device)
                    v_w = float(world_value(world(obs, act), hx, hy, target, scale))
                nt1, nt2, _w, pain_a, load_a = env.step_body(t1, t2, d1, d2)
                ahx, ahy = env.forward_kinematics(nt1, nt2)
                v_t = (-math.hypot(ahx - target[0], ahy - target[1])
                       - 5.0 * (math.exp(min(pain_a, env.PAIN_CAP) / 10.0) - 1.0)
                       - 2.0 * load_a)
                acc_w = v_w > v_cur + ACCEPT_EPS
                acc_t = v_t > v_cur + ACCEPT_EPS
                s = stat[key]
                s["n"] += 1
                s["abs_err"] += abs(v_w - v_t)
                if acc_t and not acc_w:
                    s["err_rej"] += 1
                if acc_w and not acc_t:
                    s["err_acc"] += 1

    print(f"状态样本 {args.samples} | 每个状态 {len(ACTIONS) ** 2} 个候选动作 | base={args.base}")
    print(f"{'动作大小':>10} | {'样本':>7} | {'平均|ΔV|误差':>12} | {'误拒率':>8} | {'误接率':>8}")
    for key, _, _ in BUCKETS:
        s = stat[key]
        n = max(1, s["n"])
        print(f"{key:>10} | {s['n']:>7} | {s['abs_err'] / n:>12.3f} | "
              f"{s['err_rej'] / n:>8.3%} | {s['err_acc'] / n:>8.3%}")
    print()
    print("误拒 = 真能变好却被拒绝（世界模型太保守/偏错）")
    print("误接 = 真变差却被放行（世界模型太乐观/偏错）")


if __name__ == "__main__":
    main()
