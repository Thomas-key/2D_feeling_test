#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_day.py — 白天（真实）：base 预训练（全量）+ 任务循环（LoRA 在线更新）

循环本体在 loop.run_task()——和 closed_loop_exec.py 是同一份（以前抄了两份，
一份有内层重试一份没有，导致两边 step 不是同一个时钟）。

一次任务：
  for step:  for retry(拒绝 → 更新决策 → 原地再试)  直至通过
             通过 → 执行（真实）→ 更新世界模型（门控）

用法：
  python tool/make_tasks.py --seed 1 --draws 100 --out tasks/seed1.json   # 先生成固定集
  python train_day.py --device cuda --tasks-file tasks/seed1.json --out runs/s1_base.json
  python train_day.py --device cuda --tasks 100                 # 现场随机（不要用于对比）
  python train_day.py --device cuda --target -1.0 -1.0 --max-steps 200   # 单任务调试
  python train_day.py --device cuda --tasks 30 --eval-only --base night_base.pt
"""

import argparse
import json
import os
import random

import torch

import env
from lora import DecisionLoRA, WorldLoRA
from loop import load_base, obs_tensor, run_task, true_value, vstar   # noqa: F401（仍从这里可导入）


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--tasks", type=int, default=0, help="多任务数（0=用 --target）")
    p.add_argument("--target", nargs=2, type=float, default=None, help="单任务目标")
    p.add_argument("--attempts", type=int, default=1, help="每个任务尝试次数（共享模型）")
    p.add_argument("--eval-only", action="store_true", help="只评估（权重全冻结，不更新）")
    p.add_argument("--base", default="pretrain_base.pt", help="加载的 base（睡后评估用 night_base.pt）")
    p.add_argument("--t1", type=float, default=30.0, help="初始 θ1")
    p.add_argument("--t2", type=float, default=90.0, help="初始 θ2")
    p.add_argument("--max-steps", type=int, default=100, help="最多接受多少次移动")
    p.add_argument("--max-retries", type=int, default=50, help="同一个状态最多试几次")
    p.add_argument("--per-task-reset", action="store_true",
                   help="每个任务开始前把 LoRA 清零（每个任务都从 base 重新适应）")
    p.add_argument("--keep-world", action="store_true",
                   help="配合 --per-task-reset：**不**重置世界 LoRA（身体图式跨任务累积）")
    p.add_argument("--stall-limit", type=int, default=15,
                   help="连续多少步没刷新最近距离就停（100=不让它停）")
    p.add_argument("--lr", type=float, default=1e-4, help="决策 LoRA 学习率")
    p.add_argument("--world-lr", type=float, default=1e-4,
                   help="世界 LoRA 学习率（它修的是身体图式，通常需要比决策大得多）")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tasks-file", default=None,
                   help="固定的任务集（tool/make_tasks.py 生成）——对比实验必须用这个")
    p.add_argument("--out", default="day_result.json",
                   help="结果 json 路径（重复跑时别互相覆盖）")
    p.add_argument("--verbose", action="store_true", help="每步打印（单任务调试用）")
    args = p.parse_args()

    # 固定 torch 随机源：LoRA 的 A 初始化用的是全局 RNG，不固定的话同一个命令
    # 两次能差 ±2 个任务（README 局限 7）。注意：这会让历史数字不能逐位复现。
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    rng = random.Random(args.seed)
    dec = DecisionLoRA().to(device)
    world = WorldLoRA().to(device)

    # 1. 加载 base（预训练 pretrain_base / 睡后 night_base）
    if os.path.exists(args.base):
        load_base(dec, world, args.base)
        print(f"[init] base: {args.base}", flush=True)
    else:
        print(f"[warn] 无 {args.base}——base 随机！先跑 pretrain_lora.py", flush=True)

    # 2. 白天：任务循环（只训练 LoRA）
    dec_lora_params = (list(dec.loras()["shoulder"].parameters())
                       + list(dec.loras()["elbow"].parameters()))
    world_lora_params = list(world.loras_dict().parameters())
    opt_d = torch.optim.Adam(dec_lora_params, lr=args.lr)
    opt_w = torch.optim.Adam(world_lora_params, lr=args.world_lr)

    # 任务列表（--tasks-file 固定集 / --target 单任务 / --tasks 现场随机）
    tasks = []
    if args.tasks_file:
        with open(args.tasks_file, encoding="utf-8") as f:
            payload = json.load(f)
        raw = payload["tasks"] if isinstance(payload, dict) else payload
        tasks = [tuple(t) for t in raw if env.inverse_kinematics(*t) is not None]
        print(f"[tasks] {args.tasks_file}: {len(tasks)} 个任务"
              f"（文件里 {len(raw)} 个）", flush=True)
    elif args.target is not None:
        tasks = [tuple(args.target)]
    else:
        for _ in range(args.tasks):
            tg = (rng.uniform(-1.8, 1.8), rng.uniform(-1.8, 1.8))
            if env.inverse_kinematics(*tg) is not None:
                tasks.append(tg)
    attempts_list = [tg for tg in tasks for _ in range(max(1, args.attempts))]
    verbose = args.verbose or len(attempts_list) == 1

    failures, results = [], []
    successes = dec_up = wm_up = 0
    last = None
    for task, target in enumerate(attempts_list):
        if env.inverse_kinematics(*target) is None:
            continue
        if args.per_task_reset:      # 每个任务都从同一份 base 重新适应
            for lora in dec.loras().values():
                lora.reset_trainable()   # 注意不是 reset()：全零会让梯度恒为零
            if not args.keep_world:
                for lora in world.loras_dict().values():
                    lora.reset_trainable()
            opt_d = torch.optim.Adam(dec_lora_params, lr=args.lr)
            opt_w = torch.optim.Adam(world_lora_params, lr=args.world_lr)
        res = run_task(dec, world, opt_d, opt_w, target, args.t1, args.t2,
                       max_steps=args.max_steps, max_retries=args.max_retries,
                       stall_limit=args.stall_limit,
                       eval_only=args.eval_only, verbose=verbose)
        last = res
        dec_up += res["dec_up"]
        wm_up += res["wm_up"]
        reachable = vstar(target) > -0.1     # 可达（容忍 0.1——V* > -0.1）
        results.append({"target": list(target), "reachable": reachable,
                        "hit": bool(res["hit"]), "dist": res["dist"],
                        "steps": res["steps"], "dec_up": res["dec_up"],
                        "gave_up": res["gave_up"]})
        if res["hit"]:
            successes += 1
        else:
            failures.append({"target": list(target),
                             "t1": round(res["t1"], 1), "t2": round(res["t2"], 1),
                             "dist": res["dist"], "vstar": round(vstar(target), 3),
                             "gave_up": res["gave_up"]})
        if (task + 1) % 10 == 0:
            print(f"[day] task {task+1} | 成功 {successes} | 失败 {len(failures)} | "
                  f"决策更新 {dec_up} | 世界更新 {wm_up}", flush=True)

    # 3. 保存（LoRA / 失败记录 / 轨迹）
    torch.save({"dec_base": dec.base.state_dict(),
                "dec_heads": {k: getattr(dec, k).state_dict()
                              for k in ("head_shoulder", "head_elbow")},
                "dec_lora": {k: v.state_dict() for k, v in dec.loras().items()},
                "world_base": world.base.state_dict(),
                "world_heads": world.heads.state_dict(),
                "world_lora": {k: v.state_dict()
                               for k, v in world.loras_dict().items()}},
               "lora_adapt.pt" if args.per_task_reset else "lora_day.pt")
    reach = [r for r in results if r["reachable"]]
    reach_hit = sum(1 for r in reach if r["hit"])
    print(f"[completion] 可达样本 {len(reach)}/{len(results)} | 成功 {reach_hit}/{len(reach)} = "
          f"{reach_hit/max(1,len(reach)):.3f}", flush=True)
    if os.path.dirname(os.path.abspath(args.out)):
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "eval_only": args.eval_only,
                   "tasks_file": args.tasks_file, "results": results},
                  f, ensure_ascii=False)
    if last is not None:
        with open("trajectory.json", "w", encoding="utf-8") as f:
            json.dump({"target": list(last["target"]), "hit": bool(last["hit"]),
                       "traj": last["traj"]}, f, ensure_ascii=False)
    with open("failures.json", "w", encoding="utf-8") as f:
        json.dump(failures, f, ensure_ascii=False, indent=1)
    print(f"[save] {args.out} | "
          f"{'lora_adapt.pt' if args.per_task_reset else 'lora_day.pt'}"
          f"（{successes} 成功）| failures.json（{len(failures)} 失败）", flush=True)


if __name__ == "__main__":
    main()
