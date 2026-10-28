#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_stall.py — 失败任务为什么过不去？把世界模型的预测拆开看

初版只分了"一步能不能到"，但实测发现决策的动作几乎都在改善（真实 ΔV 多为正），
而世界模型几乎一律预测为变差——所以关键是：**它到底错在哪个头**。

每行给出：预测 vs 真实的 距离 / 痛 / 负载，以及 ΔV 的世界值/真实值。
最后统计三个头各自的平均偏差（正数=预测偏大，悲观/乐观一眼看出来）。

用法：
  python train_day.py --device cuda --tasks 50 --eval-only --base pretrain_base.pt
  python tool/check_stall.py --device cuda --base pretrain_base.pt
"""

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import env
from lora import DecisionLoRA, WorldLoRA
from loop import ACCEPT_EPS, load_base, obs_tensor, true_value, world_value

GRID = (-60, -30, -10, -2, 0, 2, 10, 30, 60)


def v_of(dist, pain, load):
    return -dist - 5.0 * (math.exp(min(pain, env.PAIN_CAP) / 10.0) - 1.0) - 2.0 * load


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--base", default="pretrain_base.pt")
    p.add_argument("--failures", default="failures.json")
    args = p.parse_args()

    device = torch.device(args.device)
    dec = DecisionLoRA().to(device)
    world = WorldLoRA().to(device)
    load_base(dec, world, args.base)
    scale = torch.tensor(env.OUT_SCALE, device=device)

    with open(args.failures, encoding="utf-8") as f:
        fails = json.load(f)

    print(f"base={args.base}  失败状态 {len(fails)} 个")
    print("列：当前 dist | 动作 | 预测(距离/痛/负载) | 真实(距离/痛/负载) | ΔV 世界 / 真实\n")

    acc = {"dd": 0.0, "dp": 0.0, "dl": 0.0, "dw": 0.0, "dt": 0.0, "n": 0, "one": 0, "reach": 0}
    for fail in fails:
        target = tuple(fail["target"])
        t1, t2 = fail["t1"], fail["t2"]
        hx, hy = env.forward_kinematics(t1, t2)
        dist = math.hypot(hx - target[0], hy - target[1])
        v_cur = true_value(t1, t2, target)

        obs = obs_tensor(t1, t2, target, device)
        with torch.no_grad():
            d1, d2 = dec(obs)
            act = torch.cat([d1, d2], dim=-1) / 60.0
            pred = world(obs, act)
            pn = pred * scale
            px, py = hx + float(pn[0, 0]), hy + float(pn[0, 1])
            pred_dist = math.hypot(px - target[0], py - target[1])
            pred_pain = float(pn[0, 3])
            pred_load = float(pn[0, 4])
            v_w = float(world_value(pred, hx, hy, target, scale))
        d1f, d2f = float(d1), float(d2)

        nt1, nt2, _w, pain_a, load_a = env.step_body(t1, t2, d1f, d2f)
        ahx, ahy = env.forward_kinematics(nt1, nt2)
        true_dist = math.hypot(ahx - target[0], ahy - target[1])
        v_t = v_of(true_dist, pain_a, load_a)

        best = 9.0
        for g1 in GRID:
            for g2 in GRID:
                b1, b2 = env.forward_kinematics(*env.step_body(t1, t2, g1, g2)[:2])
                best = min(best, math.hypot(b1 - target[0], b2 - target[1]))

        acc["dd"] += pred_dist - true_dist
        acc["dp"] += pred_pain - pain_a
        acc["dl"] += pred_load - load_a
        acc["dw"] += v_w - v_cur
        acc["dt"] += v_t - v_cur
        acc["n"] += 1
        acc["one"] += 1 if best < 0.1 else 0
        acc["reach"] += 1 if true_dist < 0.1 else 0

        print(f"d={dist:>4.2f} | ({d1f:>5.1f},{d2f:>5.1f}) | "
              f"{pred_dist:>5.2f}/{pred_pain:>4.1f}/{pred_load:>4.2f} | "
              f"{true_dist:>5.2f}/{pain_a:>4.1f}/{load_a:>4.2f} | "
              f"{v_w - v_cur:>7.3f} / {v_t - v_cur:>6.3f}")

    n = max(1, acc["n"])
    print()
    print(f"平均偏差（预测 − 真实）：距离 {acc['dd'] / n:+.3f} 米 | "
          f"痛 {acc['dp'] / n:+.2f} | 负载 {acc['dl'] / n:+.3f}")
    print(f"平均 ΔV：世界 {acc['dw'] / n:+.3f} | 真实 {acc['dt'] / n:+.3f} | "
          f"偏差 {acc['dw'] / n - acc['dt'] / n:+.3f}")
    print(f"决策这一动作真能到（<0.1m）：{acc['reach']}/{n}；"
          f"该状态一步能到（枚举 9×9）：{acc['one']}/{n}")
    print()
    print("读法：距离偏差为正=预测比实际远（悲观）；痛/负载偏差为正=多预测了（更保守）")
    print(f"验证门：世界 ΔV 需要 > {ACCEPT_EPS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
