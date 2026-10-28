#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_day.py — 白天（真实）：base 预训练（全量）+ 任务循环（LoRA 更新）

流程：
  1. 预训练 base（决策：规则世界想象；世界：规则数据监督——全量）
  2. 白天任务循环：目标随机 → 决策（LoRA）→ 世界（LoRA）→ 验证-执行
     - 决策更新：拒绝时（价值梯度）
     - 世界更新：执行后（门控 MSE）
     - 成功 → 保存 LoRA（lora_success.pt）
     - 失败 → 记录（目标 + 最后状态 → failures.json）

用法：
  python train_day.py --device cuda --tasks 100 --pretrain-steps 5000
"""

import argparse
import json
import math
import os
import random

import torch
import torch.nn.functional as F

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


def vstar(target):
    """V*：可达 → 0；不可达 → -(d-2)（最近可达点）。"""
    d = math.hypot(target[0], target[1])
    return 0.0 if d <= 2.0 else -(d - 2.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--tasks", type=int, default=0, help="多任务数（0=用 --target）")
    p.add_argument("--target", nargs=2, type=float, default=None, help="单任务目标")
    p.add_argument("--attempts", type=int, default=1, help="每个任务尝试次数")
    p.add_argument("--eval-only", action="store_true", help="只评估（不训练/不更新）")
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)   # LoRA（base 冻结——可大）
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device(args.device)
    rng = random.Random(args.seed)
    dec = DecisionLoRA().to(device)
    world = WorldLoRA().to(device)

    # 2. 白天：任务循环（LoRA 更新）
    dec_lora_params = list(dec.loras()["shoulder"].parameters()) + list(dec.loras()["elbow"].parameters())
    world_lora_params = list(world.loras_dict().parameters())
    opt_d = torch.optim.Adam(dec_lora_params, lr=args.lr)
    opt_w = torch.optim.Adam(world_lora_params, lr=args.lr)

    failures = []
    successes = 0
    results = []      # 每任务：target/可达/成功（完成度对比用）
    dec_up = 0        # 决策更新次数
    wm_up = 0         # 世界更新次数
    # 任务列表（--target 单任务 / --tasks 随机多任务）
    tasks = []
    if args.target is not None:
        tasks = [tuple(args.target)]
    else:
        for _ in range(args.tasks):
            tg = (rng.uniform(-1.8, 1.8), rng.uniform(-1.8, 1.8))
            if env.inverse_kinematics(*tg) is not None:
                tasks.append(tg)
    # 展开 attempts（每任务尝试多次）
    attempts_list = [tg for tg in tasks for _ in range(max(1, args.attempts))]
    for task, target in enumerate(attempts_list):
        if env.inverse_kinematics(*target) is None:
            continue
        # 初始状态：排除极端（θ1 ±150 内、θ2 30-150——避免折叠/反向）
        t1 = rng.uniform(-150, 150)
        t2 = rng.uniform(30, 150)
        traj = []          # 本尝试轨迹（最后保存）
        hit = False
        dist = 9.0
        for step in range(args.max_steps):
            hx, hy = env.forward_kinematics(t1, t2)
            dist = math.hypot(hx - target[0], hy - target[1])
            t1d = (t1 + 180.0) % 360.0 - 180.0        # 显示归一化
            traj.append({"step": step, "t1": round(t1d, 1), "t2": round(t2, 1),
                         "x": round(hx, 2), "y": round(hy, 2), "dist": round(dist, 2)})
            if dist < 0.1:
                hit = True
                break
            obs = obs_tensor(t1, t2, target, device)
            d1, d2 = dec(obs)
            act = torch.cat([d1, d2], dim=-1) / 60.0   # (1,2) 保留梯度
            pred = world(obs, act)
            nhx, nhy = hx + pred[0, 0], hy + pred[0, 1]          # torch（保留梯度）
            dist_t = torch.sqrt((nhx - target[0]) ** 2 + (nhy - target[1]) ** 2 + 1e-6)
            pain_t = torch.clamp(pred[0, 3] * 30.0, 0.0, 30.0)
            v_new = -dist_t - 5.0 * (torch.exp(pain_t / 10.0) - 1.0) - 2.0 * pred[0, 4]
            v_cur = true_value(t1, t2, target)
            if v_new > v_cur + 0.01:
                # 通过 → 执行 → 世界更新（eval-only 跳过更新）
                if args.eval_only:
                    t1 = t1 + float(d1)
                    t2 = min(env.FOLD_LIMIT, max(env.ELBOW_MIN, t2 + float(d2)))
                    continue
                o1, o2 = t1, t2
                t1 = t1 + float(d1)
                t2 = min(env.FOLD_LIMIT, max(env.ELBOW_MIN, t2 + float(d2)))   # 折叠 160 挡
                nhx2, nhy2 = env.forward_kinematics(t1, t2)
                ohx, ohy = env.forward_kinematics(o1, o2)
                actual = torch.tensor([nhx2 - ohx, nhy2 - ohy, 0.0,
                                       env.pain(t1, t2) / 30.0,
                                       env.load(t1, t2, None)], device=device)
                err = (pred.squeeze(0) - actual).abs()
                if float(err[:2].max()) > 0.1 or float(err[3]) > 1.0:
                    pred2 = world(obs, act.detach())
                    # 线性 MSE（位置/负载/触感）+ 痛项指数（价值一致）
                    err = (pred2.squeeze(0) - actual) ** 2
                    loss_w = err.mean()
                    pain_pred = torch.exp(torch.clamp(pred2[0, 3] * 30.0, -30.0, 60.0) / 10.0)
                    pain_act = torch.exp(torch.clamp(actual[3] * 30.0, -30.0, 60.0) / 10.0)
                    loss_w = loss_w + 0.5 * (pain_pred - pain_act) ** 2
                    opt_w.zero_grad()
                    loss_w.backward()
                    opt_w.step()
                    wm_up += 1
            else:
                # 拒绝 → 决策更新（LoRA）
                loss_d = -v_new
                opt_d.zero_grad()
                loss_d.backward()
                opt_d.step()
                dec_up += 1
        reachable = vstar(target) > -0.1     # 可达（容忍 0.1——V* > -0.1）
        results.append({"target": list(target), "reachable": reachable,
                        "hit": bool(hit), "dist": round(dist, 3)})
        if hit:
            successes += 1
        else:
            failures.append({"target": list(target),
                             "t1": round(t1, 1), "t2": round(t2, 1),
                             "dist": round(dist, 2), "vstar": round(vstar(target), 3)})
        if (task + 1) % 10 == 0:
            print(f"[day] task {task+1} | 成功 {successes} | 失败 {len(failures)} | "
                  f"决策更新 {dec_up} | 世界更新 {wm_up}", flush=True)

    # 3. 保存（成功 LoRA / 失败记录）
    torch.save({"dec_base": dec.base.state_dict(),
                "dec_heads": {k: getattr(dec, k).state_dict()
                              for k in ("head_shoulder", "head_elbow")},
                "dec_lora": {k: v.state_dict() for k, v in dec.loras().items()},
                "world_base": world.base.state_dict(),
                "world_heads": world.heads.state_dict(),
                "world_lora": {k: v.state_dict()
                               for k, v in world.loras_dict().items()}}, "lora_day.pt")
    # 完成度（可达样本成功率）
    reach = [r for r in results if r["reachable"]]
    reach_hit = sum(1 for r in reach if r["hit"])
    print(f"[completion] 可达样本 {len(reach)}/{len(results)} | 成功 {reach_hit}/{len(reach)} = "
          f"{reach_hit/max(1,len(reach)):.3f}", flush=True)
    with open("day_result.json", "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "eval_only": args.eval_only, "results": results},
                  f, ensure_ascii=False)
    with open("trajectory.json", "w", encoding="utf-8") as f:
        json.dump({"target": list(attempts_list[-1]) if attempts_list else [0, 0],
                   "hit": bool(hit), "traj": traj}, f, ensure_ascii=False)
    with open("failures.json", "w", encoding="utf-8") as f:
        json.dump(failures, f, ensure_ascii=False, indent=1)
    print(f"[save] lora_day.pt（{successes} 成功）| failures.json（{len(failures)} 失败）", flush=True)


if __name__ == "__main__":
    main()
