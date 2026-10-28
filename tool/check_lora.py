#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_lora.py — 白天攒下的 LoRA 到底把模型改了多少？（全局影响，不是单点）

背景：把 lora_day 合并进 base 后，46 个任务里有 11 个从成功变失败。
在动手改训练之前，先量清楚是"决策 LoRA"还是"世界 LoRA"在捣乱，
以及扰动的量级（动作差几度 / 位置差几米 / 痛差多少）。

用法：
  python tool/check_lora.py --device cuda --base pretrain_base.pt --lora lora_day.pt
"""

import argparse
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import env
from lora import DecisionLoRA, WorldLoRA
from loop import load_base, obs_tensor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--base", default="pretrain_base.pt")
    p.add_argument("--lora", default="lora_day.pt")
    p.add_argument("--samples", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device(args.device)
    rng = random.Random(args.seed)
    dec = DecisionLoRA().to(device)
    world = WorldLoRA().to(device)
    load_base(dec, world, args.base, dec_lora=False, world_lora=False)   # 只读 base
    scale = torch.tensor(env.OUT_SCALE, device=device)

    states = []
    for _ in range(args.samples):
        t1 = rng.uniform(-180.0, 180.0)
        t2 = rng.uniform(env.ELBOW_MIN, env.FOLD_MAX)
        target = (rng.uniform(-1.8, 1.8), rng.uniform(-1.8, 1.8))
        states.append((t1, t2, target))

    # 1) 基准行为（LoRA = 0）
    with torch.no_grad():
        base_act = [torch.cat(dec(obs_tensor(t1, t2, tg, device)), dim=-1)[0]
                    for t1, t2, tg in states]
        base_pred = [world(obs_tensor(t1, t2, tg, device),
                           torch.tensor([[0.5, -0.3]], device=device))[0]
                     for t1, t2, tg in states]

    # 2) 载入白天的 LoRA
    ckpt = torch.load(args.lora, map_location="cpu")
    for k, lora in dec.loras().items():
        lora.load_state_dict(ckpt["dec_lora"][k])
    for k, lora in world.loras_dict().items():
        lora.load_state_dict(ckpt["world_lora"][k])

    with torch.no_grad():
        new_act = [torch.cat(dec(obs_tensor(t1, t2, tg, device)), dim=-1)[0]
                   for t1, t2, tg in states]
        new_pred = [world(obs_tensor(t1, t2, tg, device),
                          torch.tensor([[0.5, -0.3]], device=device))[0]
                    for t1, t2, tg in states]

    # 3) 权重层面的相对扰动
    print(f"=== 权重相对扰动 ||ΔW||/||W|| ===")
    for name, lora in list(dec.loras().items()) + list(world.loras_dict().items()):
        dw = (lora.B @ lora.A.T) * lora.scale
        rel = float(dw.norm() / (lora.base.weight.norm() + 1e-9))
        print(f"  {name:>8} | ||ΔW|| = {float(dw.norm()):.4f}  ||W|| = {float(lora.base.weight.norm()):.4f}"
              f"  → {rel:.1%}")

    # 4) 行为层面的扰动
    da = torch.stack(new_act) - torch.stack(base_act)
    print(f"\n=== 决策：动作变化（{args.samples} 个随机状态，单位：度）===")
    print(f"  d1 平均 |Δ| {float(da[:, 0].abs().mean()):.2f}  最大 {float(da[:, 0].abs().max()):.2f}")
    print(f"  d2 平均 |Δ| {float(da[:, 1].abs().mean()):.2f}  最大 {float(da[:, 1].abs().max()):.2f}")

    dp = (torch.stack(new_pred) - torch.stack(base_pred)) * scale
    print(f"\n=== 世界：预测变化（同一批状态，动作固定 (30°,-18°)）===")
    print(f"  位置平均 |Δ| {float(dp[:, :2].abs().mean()):.3f} 米  最大 {float(dp[:, :2].abs().max()):.3f}")
    print(f"  痛平均 |Δ|   {float(dp[:, 3].abs().mean()):.2f}      最大 {float(dp[:, 3].abs().max()):.2f}")
    print(f"  负载平均 |Δ| {float(dp[:, 4].abs().mean()):.2f}      最大 {float(dp[:, 4].abs().max()):.2f}")
    print("\n参考：目标附近一步只改变 0.1~0.2 米；验证门阈值是 0.2 米 / 3 分痛")
    return 0


if __name__ == "__main__":
    sys.exit(main())
