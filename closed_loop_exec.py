#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2d感觉实验/closed_loop_exec.py — 推理时在线更新（想象-验证-更新-执行）

每步：
  决策 → 动作
  世界模型 → 预测 → 价值
  价值通过（≥ 当前 - 0.1）→ 执行
  不通过 → 更新决策梯度（在线，lr 小）→ 重试（同一状态）
  直到通过（最多 50 次）→ 执行

用法：
  python closed_loop_exec.py --decision decision_exec_best.pt --world world_num.pt \
      --target 1.0 1.0 --device cuda
"""

import argparse
import json
import math

import torch
import torch.nn.functional as F

import env
from train_world_num import WorldNum
from train_imagine_num import DecisionNum, value_from_delta


def true_value(t1, t2, target):
    hx, hy = env.forward_kinematics(t1, t2)
    dist = math.hypot(hx - target[0], hy - target[1])
    p = env.pain(t1, t2)
    return -dist - 5.0 * (math.exp(min(p, 30.0) / 10.0) - 1.0) - 2.0 * env.load(t1, t2, None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--decision", default=None,
                   help="显式决策（不传=自动：best 优先，否则 decision_pre.pt）")
    p.add_argument("--world", default="world_num.pt")
    p.add_argument("--best-decision", default="decision_best.pt")
    p.add_argument("--best-world", default="world_best.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--target", nargs=2, type=float, default=[1.0, 1.0])
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--t1", type=float, default=30.0)
    p.add_argument("--t2", type=float, default=90.0)
    p.add_argument("--lr", type=float, default=1e-5)   # 真实环境：小 lr
    p.add_argument("--out", default="trajectory_exec.json")
    args = p.parse_args()

    device = torch.device(args.device)
    import os
    _wpath = args.best_world if os.path.exists(args.best_world) else args.world
    world = WorldNum().to(device)
    ckpt = torch.load(_wpath, map_location="cpu")
    world.load_state_dict(ckpt["model"])
    for prm in world.parameters():
        prm.requires_grad = True     # 在线动态更新
    world_opt = torch.optim.Adam(world.parameters(), lr=1e-6)
    model = DecisionNum().to(device)
    if args.decision is not None:          # 显式传入 → 用它
        _dpath = args.decision
    elif os.path.exists(args.best_decision):
        _dpath = args.best_decision
    else:
        _dpath = "decision_pre.pt"
    model.load_state_dict(torch.load(_dpath, map_location="cpu")["model"])
    print(f"[init] 决策: {_dpath}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    target = (args.target[0], args.target[1])
    t1, t2 = args.t1, args.t2
    traj = []
    hit = False
    total_retries = 0
    dec_updates = 0
    wm_updates = 0
    for step in range(args.steps):
        hx, hy = env.forward_kinematics(t1, t2)
        dist = math.hypot(hx - target[0], hy - target[1])
        traj.append({"step": step, "t1": round(t1, 1), "t2": round(t2, 1),
                     "x": round(hx, 2), "y": round(hy, 2), "dist": round(dist, 2)})
        if dist < 0.1:
            hit = True
            break
        v_cur = true_value(t1, t2, target)
        ok = False
        for retry in range(300):
            obs = torch.tensor(env.obs_vector(t1, t2, target), dtype=torch.float32,
                               device=device).unsqueeze(0)
            action = model(obs)
            act = action / 60.0
            pred = world(obs, act)
            v_new = value_from_delta(pred.squeeze(0), hx, hy, target)
            if float(v_new) > v_cur + 0.01:     # 必须改善（不动不通过）
                d1, d2 = float(action[0, 0]), float(action[0, 1])
                o1, o2 = t1, t2
                t1 = t1 + d1
                t2 = min(env.ELBOW_MAX, max(env.ELBOW_MIN, t2 + d2))
                # 在线世界模型更新（执行后对比预测 vs 实际）
                nhx, nhy = env.forward_kinematics(t1, t2)
                ohx, ohy = env.forward_kinematics(o1, o2)
                actual = torch.tensor([nhx - ohx, nhy - ohy, 0.0,
                                       env.pain(t1, t2) / 30.0,
                                       env.load(t1, t2, None)], device=device)
                err = (pred.squeeze(0) - actual).abs()
                if float(err[:2].max()) > 0.1 or float(err[3]) > 1.0:
                    pred2 = world(obs, act.detach())
                    loss_wm = F.mse_loss(pred2.squeeze(0), actual)
                    world_opt.zero_grad()
                    loss_wm.backward()
                    world_opt.step()
                    wm_updates += 1
                ok = True
                break
            # 拒绝 → 在线更新决策（想象）
            loss = -v_new
            opt.zero_grad()
            loss.backward()
            opt.step()
            dec_updates += 1
            total_retries += 1
        if not ok:
            # 兜底：强制执行（不卡死）
            d1, d2 = float(action[0, 0]), float(action[0, 1])
            t1 = t1 + d1
            t2 = min(env.ELBOW_MAX, max(env.ELBOW_MIN, t2 + d2))
    hx, hy = env.forward_kinematics(t1, t2)
    dist = math.hypot(hx - target[0], hy - target[1])
    if hit:
        torch.save({"model": model.state_dict()}, args.best_decision)
        torch.save({"model": world.state_dict()}, args.best_world)
        print(f"[best] 达标！保存 {args.best_decision} / {args.best_world}", flush=True)
    print(f"[闭环] 步数 {len(traj)} | 到达 {hit} | 最终距离 {dist:.2f} | "
          f"总重试 {total_retries} | 决策更新 {dec_updates} | 世界更新 {wm_updates}")
    print(f"[轨迹] 前 5: {traj[:5]}")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"target": list(target), "hit": hit, "traj": traj}, f, ensure_ascii=False)
    print(f"[save] {args.out}", flush=True)


if __name__ == "__main__":
    main()
