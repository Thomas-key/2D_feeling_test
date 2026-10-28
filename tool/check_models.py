#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_models.py — 单独测试决策/世界预测（加载 pretrain_base.pt）

用法：
  python tool/check_models.py --device cuda
"""

import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import env  # noqa: E402
from lora import DecisionLoRA, WorldLoRA  # noqa: E402


def obs_tensor(t1, t2, target, device):
    return torch.tensor(env.obs_vector(t1, t2, target), dtype=torch.float32,
                        device=device).unsqueeze(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--base", default="pretrain_base.pt")
    args = p.parse_args()
    device = torch.device(args.device)
    dec = DecisionLoRA().to(device)
    world = WorldLoRA().to(device)
    ckpt = torch.load(args.base, map_location="cpu")
    dec.base.load_state_dict(ckpt["dec_base"])
    for k in ("head_shoulder", "head_elbow"):
        getattr(dec, k).load_state_dict(ckpt["dec_heads"][k])
    world.base.load_state_dict(ckpt["world_base"])
    world.heads.load_state_dict(ckpt["world_heads"])
    dec.eval()
    world.eval()

    # ==== 决策测试：方向是否朝目标 ====
    print("=== 决策测试（动作是否朝目标）===")
    print(f"{'状态(θ1,θ2)':>14} | {'目标':>10} | {'手位置':>12} | {'决策(d1,d2)':>12} | {'目标方向':>8}")
    cases = [(0, 90, (1.0, 1.0)), (90, 90, (-1.0, 1.0)), (180, 90, (-1.0, -1.0)),
             (-90, 90, (1.0, -1.0)), (0, 90, (-1.0, 0.0)), (0, 90, (2.0, 0.0))]
    for t1, t2, target in cases:
        obs = obs_tensor(t1, t2, target, device)
        with torch.no_grad():
            d1, d2 = dec(obs)
        hx, hy = env.forward_kinematics(t1, t2)
        ang = math.degrees(math.atan2(target[1] - hy, target[0] - hx))
        print(f"({t1:>4},{t2:>4}) | {str(target):>10} | ({hx:>5.2f},{hy:>5.2f}) | "
              f"({float(d1):>6.1f},{float(d2):>5.1f}) | {ang:>7.0f}°")

    # ==== 世界测试：预测位置 vs 实际 ====
    print("\n=== 世界测试（预测 vs 实际）===")
    print(f"{'状态':>10} | {'动作':>10} | {'预测手位置':>14} | {'实际手位置':>14} | {'误差':>6}")
    t1, t2 = 0.0, 90.0
    hx, hy = env.forward_kinematics(t1, t2)
    for d1, d2 in [(0, 0), (30, 0), (-30, 0), (0, 30), (0, -30), (60, -60), (-60, 60)]:
        obs = obs_tensor(t1, t2, (1.0, 1.0), device)
        act = torch.tensor([[d1 / 60.0, d2 / 60.0]], dtype=torch.float32, device=device)
        with torch.no_grad():
                pred = world(obs, act)
        # 反归一化（量纲还原）
        import torch as _t
        scale = _t.tensor(env.OUT_SCALE, device=pred.device)
        pred = pred * scale
        px, py = hx + float(pred[0, 0]), hy + float(pred[0, 1])
        nt1, nt2, _w2, _pa, _la = env.step_body(t1, t2, d1, d2)
        ax, ay = env.forward_kinematics(nt1, nt2)
        err = math.hypot(px - ax, py - ay)
        print(f"({t1:>3},{t2:>3}) | ({d1:>3},{d2:>3}) | ({px:>5.2f},{py:>5.2f}) | "
              f"({ax:>5.2f},{ay:>5.2f}) | {err:>5.2f}")

    # ==== 世界测试：痛预测（折叠）====
    print("\n=== 世界测试（折叠痛——θ2 到 160 被挡）===")
    for d2 in (30, 60, 90):
        t2c = 100.0
        obs = obs_tensor(0.0, t2c, (1.0, 1.0), device)
        act = torch.tensor([[0.0, d2 / 60.0]], dtype=torch.float32, device=device)
        with torch.no_grad():
            pred = world(obs, act)
        nt1, nt2, want2, pain_a, load_a = env.step_body(0.0, t2c, 0.0, d2)
        print(f"θ2=100 + 动作 {d2}° → 预测痛 {float(pred[0, 3]) * 30:>5.1f} | "
              f"实际 θ2={nt2:.0f}（想推 {want2:.0f}）痛 {pain_a:>4.1f} | 负载 {load_a:.2f}")


if __name__ == "__main__":
    main()
