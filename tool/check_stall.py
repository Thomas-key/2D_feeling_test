#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_stall.py — 那些失败的任务，到底是"不会走"还是"走得动却被门拦了"？

背景：基线只有 50% 成功。夜间失败状态算出的"当前价值"经常是 -0.001，
看起来决策下一步就能贴到目标——但白天它就是过不去。
这个脚本把每个失败状态摊开，判断三类原因：

  ① 一步就能到（真实最优 < 0.1 米），且决策选的动作也能到 → 验证门误拒
  ② 一步就能到，但决策选的动作到不了 → 决策选错（技能问题）
  ③ 一步到不了（真实最优 ≥ 0.1 米） → 技能确实缺失

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
from loop import ACCEPT_EPS, HIT_DIST, load_base, obs_tensor, true_value, world_value

GRID = (-60, -30, -10, -2, 0, 2, 10, 30, 60)


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

    try:
        with open(args.failures, encoding="utf-8") as f:
            fails = json.load(f)
    except FileNotFoundError:
        print(f"没有 {args.failures}——先跑一次 train_day 生成失败清单")
        return 2

    n_gate = n_wrong = n_skill = 0
    print(f"base={args.base}  失败状态 {len(fails)} 个\n")
    print(f"{'状态':>16} | {'目标':>16} | {'dist':>5} | {'决策动作':>12} | "
          f"{'世界ΔV':>8} | {'真实ΔV':>8} | {'真实最优一步':>10} | 判定")
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
            v_w = float(world_value(world(obs, act), hx, hy, target, scale))
        d1f, d2f = float(d1), float(d2)

        # 决策这个动作，真实执行后会怎样
        nt1, nt2, _w, pain_a, load_a = env.step_body(t1, t2, d1f, d2f)
        ahx, ahy = env.forward_kinematics(nt1, nt2)
        dist_act = math.hypot(ahx - target[0], ahy - target[1])
        v_t = (-math.hypot(ahx - target[0], ahy - target[1])
               - 5.0 * (math.exp(min(pain_a, env.PAIN_CAP) / 10.0) - 1.0)
               - 2.0 * load_a)

        # 这个状态上，真实最优的一步能到多近？
        best = 9.0
        for g1 in GRID:
            for g2 in GRID:
                b1, b2 = env.forward_kinematics(*env.step_body(t1, t2, g1, g2)[:2])
                best = min(best, math.hypot(b1 - target[0], b2 - target[1]))

        dv_w, dv_t = v_w - v_cur, v_t - v_cur
        if best < HIT_DIST:                     # 一步就能到
            if dist_act < HIT_DIST and dv_w <= ACCEPT_EPS:
                blame, n_gate = "门误拒", n_gate + 1
            elif dist_act >= HIT_DIST:
                blame, n_wrong = "选错动作", n_wrong + 1
            else:
                blame = "门误拒?"
                n_gate += 1
        else:
            blame, n_skill = "技能缺失", n_skill + 1

        print(f"({t1:>6.1f},{t2:>5.1f}) | ({target[0]:>6.2f},{target[1]:>6.2f}) | {dist:>5.2f} | "
              f"({d1f:>5.1f},{d2f:>5.1f}) | {dv_w:>8.3f} | {dv_t:>8.3f} | {best:>10.2f} | {blame}")

    n = max(1, len(fails))
    print()
    print(f"门误拒（一步能到、决策也对，但世界ΔV 没过线）：{n_gate}/{len(fails)} = {n_gate/n:.0%}")
    print(f"选错动作（一步能到，但决策挑的动作到不了）：    {n_wrong}/{len(fails)} = {n_wrong/n:.0%}")
    print(f"技能缺失（这个状态一步到不了目标）：            {n_skill}/{len(fails)} = {n_skill/n:.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
