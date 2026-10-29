#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_world.py — 世界模型到底准不准？（比"改了多少"重要）

为什么需要：
  冻结评估（--eval-only）**看不见世界模型**。冻结模式下每个状态只问决策一次，
  而"接受"和"没接受"执行的都是同一个动作——世界模型唯一的影响是安全门。
  实测：一天累积的世界 LoRA 与不训练在 180 个任务上逐位相同（0 胜 0 负）。
  所以"世界有没有变准"必须单独量。

量四件事（同一批样本，所有检查点共用 → 可以做配对）：
  位置误差     预测手位置 vs 真实手位置（米）
  痛误差       预测痛 vs 真实痛（分）
  价值误差     |预测 V' - 真实 V'|——验证门真正比较的是这个
  判定翻转     预测"这步值得走吗"与真值结论不一致的比例（决定会不会走错）

⚠️ 配对很重要：同一批样本上，一个检查点的误差降了 0.004 到底算不算数，
   要看不一致方向（更好/更差多少个样本），而不是只看均值。

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
    """返回逐样本误差列表（配对比较要用）。"""
    dec = DecisionLoRA().to(device)
    world = WorldLoRA().to(device)
    load_base(dec, world, path, dec_lora=False, world_lora=True)   # 世界 LoRA 一起加载
    world.eval()
    scale = torch.tensor(env.OUT_SCALE, device=device)

    pos_e, pain_e, load_e, val_e, flip = [], [], [], [], []
    false_ok = 0
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

            pos_e.append(math.hypot(phx - nhx, phy - nhy))
            pain_e.append(abs(p_pred - pain_a))
            load_e.append(abs(l_pred - load_a))

            v_cur = true_value(t1, t2, target)
            v_pred = value_of(phx, phy, target, p_pred, l_pred)
            v_true = value_of(nhx, nhy, target, pain_a, load_a)
            val_e.append(abs(v_pred - v_true))
            ok_pred = v_pred > v_cur + ACCEPT_EPS
            ok_true = v_true > v_cur + ACCEPT_EPS
            flip.append(1 if ok_pred != ok_true else 0)
            if ok_pred and not ok_true:
                false_ok += 1
    return {"pos": pos_e, "pain": pain_e, "load": load_e, "value": val_e,
            "flip": flip, "false_ok": false_ok}


def mean(xs):
    return sum(xs) / max(1, len(xs))


def sign_p(better, worse):
    """双尾精确符号检验。大样本走对数空间——2000 个样本时 2**n 早就溢出了。"""
    n = better + worse
    if n == 0:
        return 1.0
    k = min(better, worse)
    if n <= 200:
        return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))
    lg = math.lgamma
    tail = 0.0
    for i in range(k + 1):
        tail += math.exp(lg(n + 1) - lg(i + 1) - lg(n - i + 1) - n * math.log(2.0))
    return min(1.0, 2.0 * tail)


def paired(base, other, key):
    """逐样本配对：均值差、标准误、更好/更差多少个、符号检验 p。"""
    a, b = base[key], other[key]
    d = [y - x for x, y in zip(a, b)]
    n = len(d)
    m = mean(d)
    var = sum((x - m) ** 2 for x in d) / max(1, n - 1)
    se = math.sqrt(var / n)
    better = sum(1 for x in d if x < -1e-12)
    worse = sum(1 for x in d if x > 1e-12)
    return m, se, better, worse, sign_p(better, worse)


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
    print(f"{'检查点':<24}{'位置(m)':>9}{'痛(分)':>8}{'负载':>7}{'价值':>8}"
          f"{'判定翻转':>9}{'误接':>7}")

    results = []
    base = None
    for path in args.ckpts:
        if not os.path.exists(path):
            print(f"{path:<24}  (不存在)")
            continue
        r = evaluate(path, samples, device)
        n = len(r["pos"])
        tag = os.path.basename(path)[:23]
        fl = sum(r["flip"]) / n
        print(f"{tag:<24}{mean(r['pos']):>9.4f}{mean(r['pain']):>8.2f}"
              f"{mean(r['load']):>7.3f}{mean(r['value']):>8.4f}"
              f"{fl:>8.1%}{r['false_ok']:>7}")
        if base is None:
            base = r
        else:
            print(f"  {'':<22}{'配对变化（更好/更差，符号检验）':<40}")
            for key, name, unit in (("pos", "位置", "m"), ("value", "价值", ""),
                                    ("pain", "痛", "分"), ("load", "负载", "")):
                m, se, better, worse, pv = paired(base, r, key)
                print(f"  {name:<6} {m:+.4f}{unit} ± {se:.4f}  "
                      f"更好 {better} / 更差 {worse}   p={pv:.3f}")
            fb = sum(base["flip"])
            fo = sum(r["flip"])
            # 判定翻转是 0/1，用配对 McNemar
            only_o = sum(1 for x, y in zip(base["flip"], r["flip"]) if y and not x)
            only_b = sum(1 for x, y in zip(base["flip"], r["flip"]) if x and not y)
            print(f"  {'翻转':<6} {fo/n:.1%} vs {fb/n:.1%}  "
                  f"变好 {only_b} / 变坏 {only_o}   p={sign_p(only_o, only_b):.3f}")
        results.append(r)

    print()
    print("怎么读：均值差要跟标准误一起看（差 < 2×SE 基本是噪声）；")
    print("        '更好/更差'个数才是配对的意义——各一半就是没变。")
    print("判定翻转 = 预测说'这步值得走'与真值结论不一致的比例；误接 = 危险方向。")
    print()
    print("⚠️ 探针分布是 dream.sample（整个状态空间）。白天的更新只发生在")
    print("   '从 (30°,90°) 出发、朝任务目标走'的窄走廊里，全局平均会把局部学习稀释掉。")


if __name__ == "__main__":
    main()
