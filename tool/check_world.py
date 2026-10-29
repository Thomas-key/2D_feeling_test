#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_world.py — 世界模型到底准不准？（比"改了多少"重要）

为什么需要：
  冻结评估（--eval-only）**看不见世界模型**。冻结模式下每个状态只问决策一次，
  而"接受"和"没接受"执行的都是同一个动作——世界模型唯一的影响是安全门。
  实测：一天累积的世界 LoRA 与不训练在 180 个任务上逐位相同（0 胜 0 负）。
  所以"世界有没有变准"必须单独量。

这里量四件事（同一批样本，所有检查点共用）：
  位置误差     预测手位置 vs 真实手位置（米）
  痛误差       预测痛 vs 真实痛（分）
  价值误差     |预测 V' - 真实 V'|——验证门真正比较的是这个
  误判率       预测的"这一步值得走吗"与真值判断不一致的比例
               （就是价值判定翻转的比例；它才决定智能体会不会走错）

用法：
  python tool/check_world.py --device cuda --samples 2000 ^
      --ckpts pretrain_base.pt runs/s1_both.pt runs/s1_world.pt
"""

import argparse
import math
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dream  # noqa: E402
import env  # noqa: E402
from lora import DecisionLoRA, WorldLoRA  # noqa: E402
from loop import ACCEPT_EPS, load_base, true_value  # noqa: E402


def make_samples(n, seed):
    """和预训练/回放同一套分布（dream.sample），固定种子 → 所有检查点看同一批样本。"""
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        s = dream.sample(rng)
        if s is not None:
            out.append(s)
    return out


def value_of(hx, hy, target, pain, load):
    """验证门用的价值（与 loop.world_value 同口径，只是用标量算）。"""
    dist = math.hypot(hx - target[0], hy - target[1])
    p = min(max(pain, 0.0), env.PAIN_CAP)
    return -dist - 5.0 * (math.exp(p / 10.0) - 1.0) - 2.0 * max(load, 0.0)


def evaluate(path, samples, device):
    dec = DecisionLoRA().to(device)
    world = WorldLoRA().to(device)
    load_base(dec, world, path, dec_lora=False, world_lora=True)   # 世界 LoRA 一起加载
    world.eval()
    scale = torch.tensor(env.OUT_SCALE, device=device)

    e_pos = e_pain = e_load = e_val = 0.0
    e_pain_zone = 0.0
    pain_zone_n = 0
    bad = 0                        # 价值判定翻转次数
    false_ok = 0                   # 真值说"不值"，预测说"值"（危险方向）
    n = len(samples)
    with torch.no_grad():
        for t1, t2, target, d1r, d2r in samples:
            obs = torch.tensor(env.obs_vector(t1, t2, target), dtype=torch.float32,
                               device=device).unsqueeze(0)
            act = torch.tensor([[d1r / 60.0, d2r / 60.0]], device=device)
            pred = (world(obs, act) * scale)[0]

            n1, n2, _w, pain_a, load_a = env.step_body(t1, t2, d1r, d2r)
            ohx, ohy = env.forward_kinematics(t1, t2)
            nhx, nhy = env.forward_kinematics(n1, n2)

            p_pred = min(max(float(pred[3]), 0.0), env.PAIN_CAP)   # 与 world_value 一样夹到 ≥0
            l_pred = max(float(pred[4]), 0.0)
            phx, phy = ohx + float(pred[0]), ohy + float(pred[1])

            e_pos += math.hypot(phx - nhx, phy - nhy)
            e_pain += abs(p_pred - pain_a)
            e_load += abs(l_pred - load_a)
            if pain_a > 0:
                pain_zone_n += 1
                e_pain_zone += abs(p_pred - pain_a)

            v_cur = true_value(t1, t2, target)
            v_pred = value_of(phx, phy, target, p_pred, l_pred)
            v_true = value_of(nhx, nhy, target, pain_a, load_a)
            e_val += abs(v_pred - v_true)
            ok_pred = v_pred > v_cur + ACCEPT_EPS
            ok_true = v_true > v_cur + ACCEPT_EPS
            if ok_pred != ok_true:
                bad += 1
                if ok_pred and not ok_true:
                    false_ok += 1
    return {"n": n, "pos": e_pos / n, "pain": e_pain / n, "load": e_load / n,
            "value": e_val / n, "bad": bad, "false_ok": false_ok,
            "pain_zone_n": pain_zone_n,
            "pain_zone": e_pain_zone / max(1, pain_zone_n)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--samples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--ckpts", nargs="+", default=["pretrain_base.pt"])
    args = p.parse_args()
    device = torch.device(args.device)
    samples = make_samples(args.samples, args.seed)
    pain_n = sum(1 for t1, t2, _tg, d1, d2 in samples
                 if env.step_body(t1, t2, d1, d2)[3] > 0)
    print(f"样本 {len(samples)} 个（dream.sample，seed {args.seed}）；"
          f"其中真值会痛（痛 > 0）的 {pain_n} 个")
    print()
    print(f"{'检查点':<26}{'位置(m)':>9}{'痛(分)':>8}{'负载':>7}{'价值':>8}"
          f"{'判定翻转':>10}{'误接':>7}")
    base = None
    for path in args.ckpts:
        if not os.path.exists(path):
            print(f"{path:<26}  (不存在)")
            continue
        r = evaluate(path, samples, device)
        tag = os.path.basename(path)[:25]
        print(f"{tag:<26}{r['pos']:>9.4f}{r['pain']:>8.2f}{r['load']:>7.3f}"
              f"{r['value']:>8.4f}{r['bad']/r['n']:>9.1%}{r['false_ok']:>7}")
        if base is None:
            base = r
        else:
            print(f"{'  ↑ 相对第一个':<26}{r['pos']-base['pos']:>+9.4f}"
                  f"{r['pain']-base['pain']:>+8.2f}{r['load']-base['load']:>+7.3f}"
                  f"{r['value']-base['value']:>+8.4f}"
                  f"{(r['bad']-base['bad'])/r['n']:>+9.1%}{r['false_ok']-base['false_ok']:>+7d}")
    print()
    print("判定翻转 = 预测说'这步值得走'与真值结论不一致的比例（决定智能体会不会走错）")
    print("误接     = 真值说不值得、预测说值得的次数（危险方向）")
    print("世界 LoRA 只影响这几项；冻结任务的完成度几乎不受它影响，别用完成度评价世界模型。")


if __name__ == "__main__":
    main()
