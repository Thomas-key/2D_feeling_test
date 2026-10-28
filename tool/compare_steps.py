#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""compare_steps.py — 同一个任务，走多少步才到？（比完成率灵敏得多）

完成率在 85% 之后饱和，而且把"3 步到位"和"磨 200 次重试才到位"算成一样。
真正要问的是：**睡一觉之后，原本要反复微调才过的目标，是不是能更少步数到位。**

只看两组都成功的任务（成功与否已经用 compare_eval 比过），
比较每个任务的 steps（接受的移动次数），做配对符号检验。

用法：
  python tool/compare_steps.py day_base_s1.json day_night_s1_cycle2.json
"""

import json
import math
import sys


def load(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return {tuple(round(x, 4) for x in r["target"]): r for r in d["results"]}


def sign_test(fewer, more):
    n = fewer + more
    if n == 0:
        return 1.0
    k = min(fewer, more)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    a, b = load(sys.argv[1]), load(sys.argv[2])
    common = [t for t in a if t in b]
    both = [t for t in common if a[t]["hit"] and b[t]["hit"]]
    n = len(both)
    if n == 0:
        print("没有两组都成功的任务")
        return 2

    sa = [a[t]["steps"] for t in both]
    sb = [b[t]["steps"] for t in both]
    da = [a[t]["dist"] for t in both]
    db = [b[t]["dist"] for t in both]

    fewer = sum(1 for x, y in zip(sa, sb) if y < x)
    more = sum(1 for x, y in zip(sa, sb) if y > x)
    same = n - fewer - more

    print(f"A = {sys.argv[1]}")
    print(f"B = {sys.argv[2]}")
    print(f"两组都成功的任务：{n}（总共同任务 {len(common)}）")
    print()
    print(f"A 平均步数 {sum(sa) / n:.2f} | 中位 {sorted(sa)[n // 2]} | 最多 {max(sa)}")
    print(f"B 平均步数 {sum(sb) / n:.2f} | 中位 {sorted(sb)[n // 2]} | 最多 {max(sb)}")
    print(f"省步数 {fewer} 个 / 多步 {more} 个 / 持平 {same} 个，"
          f"符号检验 p = {sign_test(fewer, more):.4f}")
    print()
    print(f"A 平均最终距离 {sum(da) / n:.3f} 米 | B {sum(db) / n:.3f} 米")
    best = sorted(range(n), key=lambda i: sb[i] - sa[i])[:5]
    worst = sorted(range(n), key=lambda i: sb[i] - sa[i])[-3:]
    print()
    print("省得最多的 5 个任务：")
    for i in best:
        print(f"  {both[i]}: {sa[i]} → {sb[i]} 步")
    print("多花的 3 个任务：")
    for i in worst:
        print(f"  {both[i]}: {sa[i]} → {sb[i]} 步")
    return 0


if __name__ == "__main__":
    sys.exit(main())
