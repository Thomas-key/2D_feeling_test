#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2d感觉实验/train_exec.py — 验证-执行闭环（决策 → 世界模型/价值验证 → 真实执行/拒绝 → 反向更新决策）

每步：
  决策 → 动作
  世界模型（冻结）→ 预测 delta
  价值（可微）→ V(新状态)
  如果 V ≥ V(当前)（或阈值）→ 真实执行（环境更新：t1/t2 前进）
  否则 → 不执行（状态不变）
  反向更新决策（想象梯度，loss = -V_pred）
循环直到到达/步数上限。

用法：
  python train_exec.py --world world_num.pt --device cuda --episodes 2000
"""

import argparse
import math
import random

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
    p.add_argument("--world", default="world_num.pt", help="世界模型（预训练）")
    p.add_argument("--init-decision", default=None,
                   help="显式决策（不传=自动：best 优先，否则 decision_pre.pt）")
    p.add_argument("--init-world", default="", help="预训练世界（睡觉后）")
    p.add_argument("--best-decision", default="decision_best.pt", help="历史最佳决策（如有）")
    p.add_argument("--best-world", default="world_best.pt", help="历史最佳世界（如有）")
    p.add_argument("--device", default="cuda")
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-5)   # 真实环境：小 lr
    p.add_argument("--save", default="decision_exec.pt")
    args = p.parse_args()

    device = torch.device(args.device)
    world = WorldNum().to(device)
    import os as _os
    _wpath = args.best_world if _os.path.exists(args.best_world) else (args.init_world or args.world)
    ckpt = torch.load(_wpath, map_location="cpu")
    world.load_state_dict(ckpt["model"])
    for prm in world.parameters():
        prm.requires_grad = True     # 动态更新（小 lr）
    world_opt = torch.optim.Adam(world.parameters(), lr=1e-6)   # 世界：更小
    model = DecisionNum().to(device)
    import os
    if args.init_decision is not None:      # 显式传入 → 用它
        model.load_state_dict(torch.load(args.init_decision, map_location="cpu")["model"])
        print(f"[init] 指定决策: {args.init_decision}", flush=True)
    elif os.path.exists(args.best_decision):
        model.load_state_dict(torch.load(args.best_decision, map_location="cpu")["model"])
        print(f"[init] 历史最佳决策: {args.best_decision}", flush=True)
    else:
        model.load_state_dict(torch.load("decision_pre.pt", map_location="cpu")["model"])
        print("[init] 预训练决策: decision_pre.pt", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=500, gamma=0.5)

    rng = random.Random(0)
    for ep in range(args.episodes):
        target = (rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5))
        if env.inverse_kinematics(*target) is None:
            continue
        t1 = rng.uniform(-180, 180)
        t2 = rng.uniform(0, 180)
        accepted = 0
        retries = 0
        dec_updates = 0
        wm_updates = 0
        dist = 0.0
        for step in range(args.steps):
            hx, hy = env.forward_kinematics(t1, t2)
            dist = math.hypot(hx - target[0], hy - target[1])
            if dist < 0.1:
                break
            # 拒绝重试：固定 lr，只增加重试次数
            ok = False
            for retry in range(300):
                obs = torch.tensor(env.obs_vector(t1, t2, target), dtype=torch.float32,
                                   device=device).unsqueeze(0)
                action = model(obs)                      # 决策 → 动作
                act = action / 60.0
                pred = world(obs, act)                   # 世界模型 → 预测
                v_new = value_from_delta(pred.squeeze(0), hx, hy, target)
                v_cur = true_value(t1, t2, target)
                if float(v_new) > v_cur + 0.01:     # 必须改善（不动不通过）
                    d1, d2 = float(action[0, 0]), float(action[0, 1])
                    # 执行（真实）
                    o1, o2 = t1, t2
                    t1 = t1 + d1
                    t2 = min(env.ELBOW_MAX, max(env.ELBOW_MIN, t2 + d2))
                    # 实际结果 vs 预测 → 门控更新世界模型（动态）
                    nhx, nhy = env.forward_kinematics(t1, t2)
                    ohx, ohy = env.forward_kinematics(o1, o2)
                    actual = torch.tensor([nhx - ohx, nhy - ohy, 0.0,
                                           env.pain(t1, t2) / 30.0,
                                           env.load(t1, t2, None)], device=device)
                    err = (pred.squeeze(0) - actual).abs()
                    if float(err[:2].max()) > 0.1 or float(err[3]) > 1.0:
                        # 重新前向（新图——避免二次 backward）
                        pred2 = world(obs, act.detach())
                        loss_wm = F.mse_loss(pred2.squeeze(0), actual)
                        world_opt.zero_grad()
                        loss_wm.backward()
                        world_opt.step()
                        wm_updates += 1
                    ok = True
                    accepted += 1
                    break
                # 拒绝 → 更新决策（固定 lr）
                loss = -v_new
                opt.zero_grad()
                loss.backward()
                opt.step()
                dec_updates += 1
                retries += 1
            if not ok:
                break
        sched.step()
        if dist < 0.1:   # 达到目标 → 保存 best
            torch.save({"model": model.state_dict()}, args.best_decision)
            torch.save({"model": world.state_dict()}, args.best_world)
        if ep % 100 == 0:
            print(f"ep {ep} | 步数 {step+1} | 执行 {accepted} | 距离 {dist:.2f} | "
                  f"决策更新 {dec_updates} | 世界更新 {wm_updates}", flush=True)

    torch.save({"model": model.state_dict()}, args.save)
    torch.save({"model": world.state_dict()}, "world_exec.pt")   # 动态更新后的 world
    print(f"[save] {args.save}", flush=True)


if __name__ == "__main__":
    main()
