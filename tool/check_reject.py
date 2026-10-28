#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_reject.py — 验证：世界预测（拒绝）vs 实际（价值）——是否误拒"""
import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import env
from lora import DecisionLoRA, WorldLoRA

p = argparse.ArgumentParser()
p.add_argument("--device", default="cuda")
p.add_argument("--base", default="pretrain_base.pt")
p.add_argument("--target", nargs=2, type=float, default=[-1.0, -1.0])
p.add_argument("--states", nargs="+", type=float,
               default=[-10.0, 130.0, 0.0, 140.0, -10.0, 130.0],
               help="成对 t1 t2 状态")
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

target = (args.target[0], args.target[1])


def true_value(t1, t2):
    hx, hy = env.forward_kinematics(t1, t2)
    dist = math.hypot(hx - target[0], hy - target[1])
    p = env.pain(t1, t2)
    return -dist - 5.0 * (math.exp(min(p, 30.0) / 10.0) - 1.0) - 2.0 * env.load(t1, t2, None)


print(f"目标 {target}")
print(f"{'状态':>10} | {'实际位置':>12} | {'实际价值':>7} | {'决策动作':>10} | {'世界价值':>7} | {'实际价值(执行后)':>12} | {'结论':>8}")
for i in range(0, len(args.states), 2):
    t1, t2 = args.states[i], args.states[i + 1]
    v_cur = true_value(t1, t2)
    obs = torch.tensor(env.obs_vector(t1, t2, target), dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        d1, d2 = dec(obs)
        act = torch.cat([d1, d2], dim=-1) / 60.0
        pred = world(obs, act) * torch.tensor(env.OUT_SCALE, device=device)
    hx, hy = env.forward_kinematics(t1, t2)
    nhx, nhy = hx + float(pred[0, 0]), hy + float(pred[0, 1])
    pain = max(float(pred[0, 3]), 0.0)
    v_world = -math.hypot(nhx - target[0], nhy - target[1]) \
              - 5.0 * (math.exp(min(pain, 30.0) / 10.0) - 1.0) - 2.0 * float(pred[0, 4])
    # 实际执行
    nt1 = t1 + float(d1)
    nt2 = min(env.FOLD_LIMIT, max(env.ELBOW_MIN, t2 + float(d2)))
    v_actual = true_value(nt1, nt2)
    verdict = "误拒!" if (v_world <= v_cur + 0.01 and v_actual > v_cur + 0.01) else "正常"
    print(f"({t1:>4},{t2:>4}) | ({hx:>5.2f},{hy:>5.2f}) | {v_cur:>7.3f} | "
          f"({float(d1):>5.1f},{float(d2):>4.1f}) | {v_world:>7.3f} | {v_actual:>12.3f} | {verdict:>8}")
