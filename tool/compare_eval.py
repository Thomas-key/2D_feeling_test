#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""compare_eval.py — 两组"同一批任务"的完成度做配对比较（McNemar 精确检验）

为什么用配对：两组评估跑的是**同一批目标**（同 seed、同 --tasks），
所以不能当成两个独立样本比——那样标准误大得看不出 5 个点的差别。
逐任务配对后，只数"换赢"和"换输"的任务，检验力高得多。

用法：
  python tool/compare_eval.py day_base.json day_C.json
"""

import json
import math
import sys


def load(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return {tuple(round(x, 4) for x in r["target"]): bool(r["hit"]) for r in d["results"]}


def mcnemar_exact(b, c):
    """双尾精确 McNemar：b、c 是不一致对数。返回 p。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, p)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    a, b = load(sys.argv[1]), load(sys.argv[2])
    common = [t for t in a if t in b]
    if not common:
        print("两组没有共同任务（seed 或 --tasks 不一致？）")
        return 2

    both = sum(1 for t in common if a[t] and b[t])
    only_a = sum(1 for t in common if a[t] and not b[t])
    only_b = sum(1 for t in common if not a[t] and b[t])
    neither = sum(1 for t in common if not a[t] and not b[t])
    n = len(common)

    print(f"分组 A = {sys.argv[1]}")
    print(f"分组 B = {sys.argv[2]}")
    print(f"共同任务 {n}")
    print()
    print(f"{'':>16} | {'B 成功':>8} | {'B 失败':>8}")
    print(f"{'A 成功':>16} | {both:>8} | {only_a:>8}")
    print(f"{'A 失败':>16} | {only_b:>8} | {neither:>8}")
    print()
    print(f"A 成功 {both + only_a}/{n} = {(both + only_a) / n:.1%}")
    print(f"B 成功 {both + only_b}/{n} = {(both + only_b) / n:.1%}")
    print(f"B 净胜 A：{only_b - only_a:+d} 个任务（换赢 {only_b}，换输 {only_a}）")
    print(f"McNemar 精确检验 p = {mcnemar_exact(only_a, only_b):.4f}"
          f"   {'（差异显著）' if mcnemar_exact(only_a, only_b) < 0.05 else '（差异不显著）'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
