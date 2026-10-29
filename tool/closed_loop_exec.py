#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""closed_loop_exec.py — 单任务真实测试（薄壳：循环本体在 loop.run_task）

和 train_day.py 调同一个循环函数，只是这里只跑一个任务、打印每一步、到达后存 best。
差别只在"跑几个任务、输出什么"，不在循环逻辑本身。

用法：
  python tool/closed_loop_exec.py --target -1.0 -1.0 --device cuda
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lora import DecisionLoRA, WorldLoRA      # noqa: E402
from loop import load_base, run_task          # noqa: E402  （加载 base 也只有这一份）


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--target", nargs=2, type=float, default=[-1.0, -1.0])
    p.add_argument("--t1", type=float, default=30.0)
    p.add_argument("--t2", type=float, default=90.0)
    p.add_argument("--steps", type=int, default=100, help="最多接受多少次移动")
    p.add_argument("--retries", type=int, default=50, help="同一个状态最多试几次")
    p.add_argument("--stall-limit", type=int, default=15, help="连续多少步没更近就停")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--base", default="pretrain_base.pt")
    p.add_argument("--best-decision", default="decision_best.pt")
    p.add_argument("--best-world", default="world_best.pt")
    p.add_argument("--out", default="trajectory.json")
    args = p.parse_args()

    device = torch.device(args.device)
    dec = DecisionLoRA().to(device)
    world = WorldLoRA().to(device)
    load_base(dec, world, args.base)          # base + LoRA（night_base 的 world_lora 也在内）
    print(f"[init] base: {args.base}", flush=True)

    opt_d = torch.optim.Adam(list(dec.loras()["shoulder"].parameters())
                             + list(dec.loras()["elbow"].parameters()), lr=args.lr)
    opt_w = torch.optim.Adam(world.loras_dict().parameters(), lr=args.lr)

    target = (args.target[0], args.target[1])
    res = run_task(dec, world, opt_d, opt_w, target, args.t1, args.t2,
                   max_steps=args.steps, max_retries=args.retries,
                   stall_limit=args.stall_limit, verbose=True)

    if res["hit"]:
        torch.save({"dec_lora": {k: v.state_dict() for k, v in dec.loras().items()}},
                   args.best_decision)
        torch.save({"world_lora": {k: v.state_dict() for k, v in world.loras_dict().items()}},
                   args.best_world)
        print(f"[best] 达标！保存 {args.best_decision} / {args.best_world}", flush=True)
    print(f"[闭环] 步数 {len(res['traj'])} | 到达 {res['hit']} | "
          f"最终距离 {res['dist']:.2f} | 决策更新 {res['dec_up']} | 世界更新 {res['wm_up']}",
          flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"target": list(target), "hit": bool(res["hit"]), "traj": res["traj"]},
                  f, ensure_ascii=False)
    print(f"[save] {args.out}", flush=True)


if __name__ == "__main__":
    main()
