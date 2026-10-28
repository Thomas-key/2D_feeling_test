#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_body.py — 标量版 step_body 与可微版 step_theta2_t 是否一致

同一套阻力规则写了两份：一份纯 python（世界监督、真实执行用），
一份张量（决策价值里"想象"用）。两边一旦漂移，决策想象的身体就不是真身体了。
这个脚本逐点比对。

用法：python tool/check_body.py --device cuda
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--tol", type=float, default=1e-4)
    args = p.parse_args()
    device = torch.device(args.device)

    t2s = [0.0, 10.0, 40.0, 100.0, 150.0, 159.9, 160.0, 160.1, 165.0, 170.0, 179.0, 180.0]
    d2s = [-60, -30, -10, -2, 0, 2, 10, 30, 60]

    print(f"{'θ2':>7} | {'d2':>4} | {'标量θ2':>8} | {'张量θ2':>8} | {'差':>8} | "
          f"{'标量痛':>7} | {'张量痛':>7}")
    bad = 0
    for t2 in t2s:
        for d2 in d2s:
            _, nt_s, _, pain_s, _ = env.step_body(0.0, t2, 0.0, d2)
            dt = torch.tensor([[float(d2)]], device=device)
            nt_t = float(env.step_theta2_t(t2, dt))
            pain_t = float(env.pain_t(torch.tensor(nt_t, device=device)))
            diff = abs(nt_s - nt_t)
            flag = ""
            if diff > args.tol or abs(pain_s - pain_t) > args.tol:
                bad += 1
                flag = "   ← 不一致"
            print(f"{t2:>7.1f} | {d2:>4} | {nt_s:>8.2f} | {nt_t:>8.2f} | {diff:>8.4f} | "
                  f"{pain_s:>7.2f} | {pain_t:>7.2f}{flag}")

    print()
    print(f"[结果] {'通过（标量与张量逐点一致）' if bad == 0 else f'{bad} 处不一致'}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
