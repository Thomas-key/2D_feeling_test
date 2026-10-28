#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""closed_loop_exec.py — 白天测试（最新架构：DecisionLoRA/WorldLoRA + 验证-执行）

加载 pretrain_base.pt（决策/世界）；目标 → 决策 → 世界预测 → 价值验证
  通过 → 执行（真实）；拒绝 → 决策更新（LoRA，小 lr）
  执行后 → 世界更新（指数痛 loss）
到达 → 保存 best（decision_best/world_best）；轨迹 → trajectory.json

用法：
  python closed_loop_exec.py --target -1.0 -1.0 --device cuda
"""

import argparse
import json
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import env
from lora import DecisionLoRA, WorldLoRA


def obs_tensor(t1, t2, target, device):
    return torch.tensor(env.obs_vector(t1, t2, target), dtype=torch.float32,
                        device=device).unsqueeze(0)


def true_value(t1, t2, target):
    hx, hy = env.forward_kinematics(t1, t2)
    dist = math.hypot(hx - target[0], hy - target[1])
    p = env.pain(t1, t2)
    return -dist - 5.0 * (math.exp(min(p, 30.0) / 10.0) - 1.0) - 2.0 * env.load(t1, t2, None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--target", nargs=2, type=float, default=[-1.0, -1.0])
    p.add_argument("--t1", type=float, default=30.0)
    p.add_argument("--t2", type=float, default=90.0)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--base", default="pretrain_base.pt")
    p.add_argument("--best-decision", default="decision_best.pt")
    p.add_argument("--best-world", default="world_best.pt")
    p.add_argument("--out", default="trajectory.json")
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
    print(f"[init] pretrain_base: {args.base}", flush=True)

    opt_d = torch.optim.Adam(list(dec.loras()["shoulder"].parameters())
                             + list(dec.loras()["elbow"].parameters()), lr=args.lr)
    opt_w = torch.optim.Adam(world.loras_dict().parameters(), lr=args.lr)

    target = (args.target[0], args.target[1])
    t1, t2 = args.t1, args.t2
    traj = []
    hit = False
    dec_up = 0
    wm_up = 0
    for step in range(args.steps):
        hx, hy = env.forward_kinematics(t1, t2)
        dist = math.hypot(hx - target[0], hy - target[1])
        traj.append({"step": step, "t1": round((t1 + 180) % 360 - 180, 1),
                     "t2": round(t2, 1), "x": round(hx, 2), "y": round(hy, 2),
                     "dist": round(dist, 2)})
        if dist < 0.1:
            hit = True
            break
        v_cur = true_value(t1, t2, target)
        ok = False
        for retry in range(300):
            obs = obs_tensor(t1, t2, target, device)
            d1, d2 = dec(obs)
            act = torch.cat([d1, d2], dim=-1) / 60.0
            pred = world(obs, act)
            # 反归一化（torch——保留梯度，供决策更新）
            scale = torch.tensor(env.OUT_SCALE, device=device)
            pn = pred * scale
            nhx = hx + pn[0, 0]
            nhy = hy + pn[0, 1]
            pain_n = torch.clamp(pn[0, 3], 0.0, 30.0)
            v_new = (-torch.sqrt((nhx - target[0]) ** 2 + (nhy - target[1]) ** 2 + 1e-6)
                     - 5.0 * (torch.exp(pain_n / 10.0) - 1.0)
                     - 2.0 * pn[0, 4])
            if v_new > v_cur + 0.01:
                o1, o2 = t1, t2
                t1 = t1 + float(d1)
                t2 = min(env.FOLD_LIMIT, max(env.ELBOW_MIN, t2 + float(d2)))
                # 世界更新（执行后——指数痛）
                nhx2, nhy2 = env.forward_kinematics(t1, t2)
                ohx, ohy = env.forward_kinematics(o1, o2)
                raw = torch.tensor([nhx2 - ohx, nhy2 - ohy, 0.0,
                                    env.pain(t1, t2), env.load(t1, t2, None)], device=device)
                actual = raw / scale
                err = (pred.squeeze(0) - actual).abs()
                if float(err[:2].max()) > 0.1 or float(err[3]) > 0.1:
                    pred2 = world(obs, act.detach())
                    err2 = (pred2.squeeze(0) - actual) ** 2
                    loss_w = err2.mean()
                    pp = torch.exp(torch.clamp(pred2[0, 3] * 30.0, -30.0, 60.0) / 10.0)
                    pa = torch.exp(torch.clamp(actual[3] * 30.0, -30.0, 60.0) / 10.0)
                    loss_w = loss_w + 0.5 * (pp - pa) ** 2
                    opt_w.zero_grad()
                    loss_w.backward()
                    opt_w.step()
                    wm_up += 1
                ok = True
                break
            # 拒绝 → 决策更新
            loss_d = -v_new
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()
            dec_up += 1
        if not ok:
            break

    hx, hy = env.forward_kinematics(t1, t2)
    dist = math.hypot(hx - target[0], hy - target[1])
    if hit:
        torch.save({"dec_lora": {k: v.state_dict() for k, v in dec.loras().items()}},
                   args.best_decision)
        torch.save({"world_lora": {k: v.state_dict() for k, v in world.loras_dict().items()}},
                   args.best_world)
        print(f"[best] 达标！保存 {args.best_decision} / {args.best_world}", flush=True)
    print(f"[闭环] 步数 {len(traj)} | 到达 {hit} | 最终距离 {dist:.2f} | "
          f"决策更新 {dec_up} | 世界更新 {wm_up}", flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"target": list(target), "hit": hit, "traj": traj}, f, ensure_ascii=False)
    print(f"[save] {args.out}", flush=True)


if __name__ == "__main__":
    main()
