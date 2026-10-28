#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""split_eval.py — 把配对比较拆成"训练过的任务"和"没见过的任务"

为什么要拆：
  白天的 LoRA 是在前 18 个任务上攒出来的，而评估用的是同一批 46 个任务
  （前 18 个就是白天见过的，后 28 个它没见过）。
  如果增益几乎全在前 18 个里 → 是"记住了"（可能有灾难性遗忘）；
  如果后 28 个也涨 → 是真迁移。

用法：
  python tool/split_eval.py day_base2.json day_merge2.json --head 18
"""

import argparse
import json
import math


def load(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return [(tuple(round(x, 4) for x in r["target"]), bool(r["hit"])) for r in d["results"]]


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def report(title, pairs):
    both = sum(1 for a, b in pairs if a and b)
    only_a = sum(1 for a, b in pairs if a and not b)
    only_b = sum(1 for a, b in pairs if not a and b)
    n = len(pairs)
    if n == 0:
        print(f"{title}: 无样本")
        return
    print(f"{title}（{n} 个任务）")
    print(f"  A {both + only_a}/{n} = {(both + only_a) / n:.1%}"
          f" → B {both + only_b}/{n} = {(both + only_b) / n:.1%}")
    print(f"  换赢 {only_b} / 换输 {only_a}，净 {only_b - only_a:+d}，"
          f"McNemar p = {mcnemar(only_a, only_b):.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--head", type=int, default=0, help="前多少个任务是训练过的（0=不拆）")
    args = p.parse_args()

    a = load(args.a)
    b = dict(load(args.b))
    pairs = [(ha, b[t]) for t, ha in a if t in b]
    if not pairs:
        print("两组没有共同任务")
        return 2

    if args.head <= 0:
        report("全部", pairs)
    else:
        report(f"训练过的前 {args.head} 个", pairs[:args.head])
        report(f"没见过的后 {len(pairs) - args.head} 个", pairs[args.head:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
