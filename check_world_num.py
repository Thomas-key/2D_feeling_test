#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_world_num.py — 数值世界模型检查（敏感度 + 准确度 vs 实际）"""
import argparse
import torch

import env
from train_world_num import WorldNum


def predict(model, t1, t2, target, d1, d2, device):
    obs = torch.tensor(env.obs_vector(t1, t2, target), dtype=torch.float32,
                       device=device).unsqueeze(0)
    act = torch.tensor([[d1 / 60.0, d2 / 60.0]], dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = model(obs, act)
    return float(pred[0, 0]), float(pred[0, 1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--world", default="world_num.pt")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    device = torch.device(args.device)
    model = WorldNum().to(device)
    ckpt = torch.load(args.world, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    # 1) 敏感度：同状态不同动作
    t1, t2, target = 30.0, 90.0, (1.0, 1.0)
    hx, hy = env.forward_kinematics(t1, t2)
    print(f"状态 (大臂{t1:.0f}, 小臂{t2:.0f}) | 当前手 (${hx:.2f}, {hy:.2f}) | 目标 {target}")
    print("动作 → 世界模型预测点位（vs 实际）")
    for d1 in (-15, 0, 15):
        for d2 in (-15, 0, 15):
            px, py = predict(model, t1, t2, target, d1, d2, device)
            ax, ay = env.forward_kinematics(t1 + d1,
                                            min(env.ELBOW_MAX, max(env.ELBOW_MIN, t2 + d2)))
            print(f"  动作 ({d1:>3},{d2:>3}) → 预测 ({hx+px:.2f},{hy+py:.2f}) | 实际 ({ax:.2f},{ay:.2f})")
    # 2) 多状态准确度
    import random
    rng = random.Random(0)
    errs = []
    for _ in range(200):
        a1 = rng.uniform(-180, 180)
        a2 = rng.uniform(0, 180)
        tgt = (rng.uniform(-1.8, 1.8), rng.uniform(-1.8, 1.8))
        if env.inverse_kinematics(*tgt) is None:
            continue
        d1 = rng.uniform(-15, 15)
        d2 = rng.uniform(-15, 15)
        px, py = predict(model, a1, a2, tgt, d1, d2, device)
        h0x, h0y = env.forward_kinematics(a1, a2)
        ax, ay = env.forward_kinematics(a1 + d1,
                                        min(env.ELBOW_MAX, max(env.ELBOW_MIN, a2 + d2)))
        errs.append(((h0x + px - ax) ** 2 + (h0y + py - ay) ** 2) ** 0.5)
    print(f"\n200 状态平均位置误差: {sum(errs)/len(errs):.4f}")


if __name__ == "__main__":
    main()
