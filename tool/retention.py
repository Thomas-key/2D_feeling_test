#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""retention.py — 按任务顺序分块，看"加了某段经验之后，早期任务有没有被忘掉"

为什么需要：
  聚合完成率看不见遗忘。87 个任务各跑一次，早期做对的题后期根本没再考，
  所以"累积让模型偏科"这件事在总分里是隐形的。把任务按顺序切块后，
  如果**第 1 块掉了、最后一块涨了**，那才是偏科/遗忘的样子。

前提：两份结果都必须是**冻结评估**（--eval-only）。
  用"白天边跑边适应"的结果来比，看到的是在线适应能力，不是记忆。

用法：
  python tool/retention.py runs/s1_frozen.json runs/s1_accum_final.json --blocks 4
"""

import argparse
import json
import sys


def load(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return [(tuple(round(x, 4) for x in r["target"]), bool(r["hit"])) for r in d["results"]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("a", help="训练前的冻结评估（pretrain_base）")
    p.add_argument("b", help="训练后的冻结评估（一天结束的适配器）")
    p.add_argument("--blocks", type=int, default=4, help="按任务顺序切几块")
    args = p.parse_args()

    a = load(args.a)
    b = dict(load(args.b))
    seq = [(t, h) for t, h in a if t in b]
    if not seq:
        print("两组没有共同任务（任务集或顺序不一致？）")
        return 2
    n = len(seq)
    k = max(1, args.blocks)
    print(f"A = {args.a}")
    print(f"B = {args.b}")
    print(f"共同任务 {n}，按顺序切成 {k} 块")
    print()
    print(f"{'块':<14}{'n':>4}{'A 成功':>12}{'B 成功':>12}{'变化':>8}{'换赢/换输':>12}")

    sizes = [(i * n) // k for i in range(k + 1)]
    early = late = None
    for bi in range(k):
        lo, hi = sizes[bi], sizes[bi + 1]
        chunk = seq[lo:hi]
        if not chunk:
            continue
        ha = sum(1 for _, h in chunk if h)
        hb = sum(1 for t, _ in chunk if b[t])
        win = sum(1 for t, h in chunk if (not h) and b[t])
        loss = sum(1 for t, h in chunk if h and not b[t])
        delta = hb - ha
        print(f"{f'{lo+1}-{hi}':<14}{len(chunk):>4}"
              f"{f'{ha} ({ha/len(chunk):.0%})':>12}"
              f"{f'{hb} ({hb/len(chunk):.0%})':>12}"
              f"{delta:>+8}{f'+{win}/-{loss}':>12}")
        if bi == 0:
            early = delta
        late = delta

    tot_a = sum(1 for _, h in seq if h)
    tot_b = sum(1 for t, _ in seq if b[t])
    print()
    print(f"整体 {tot_a}/{n} = {tot_a/n:.1%} → {tot_b}/{n} = {tot_b/n:.1%}"
          f"（{tot_b - tot_a:+d}）")
    if early is not None and late is not None and k >= 2:
        if early < 0 <= late:
            print("⚠️ 第一块退步、最后一块进步 → 像遗忘/偏科，值得再查")
        elif early >= 0:
            print("第一块没有退步 → 没有遗忘的迹象（至少在这个粒度上）")
        else:
            print("第一块退步，但最后一块也没涨 → 像是整体变差，不是遗忘")
    return 0


if __name__ == "__main__":
    sys.exit(main())
