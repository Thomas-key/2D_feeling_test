#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""retention.py — 按任务顺序分块，看"加了某段经验之后，早期任务有没有被忘掉"

为什么需要：
  聚合完成率看不见遗忘。87 个任务各跑一次，早期做对的题后期根本没再考，
  所以"累积让模型偏科"这件事在总分里是隐形的。把任务按顺序切块后，
  如果**第 1 块显著退步、最后一块显著进步**，那才是偏科/遗忘的样子。

前提：两份结果都必须是**冻结评估**（--eval-only）。
  用"白天边跑边适应"的结果来比，看到的是在线适应能力，不是记忆。

用法：
  python tool/retention.py runs/s1_frozen.json runs/s1_accum_final.json --blocks 4

读法（重要）：
  每块只有 20 来个任务，检验力很低。不一致对里至少要 6:0 才算显著（p<0.05）。
  也就是说：一两个任务的涨跌在这里**什么都说明不了**，别把它读成遗忘。
"""

import argparse
import json
import math
import sys


def load(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return [(tuple(round(x, 4) for x in r["target"]), bool(r["hit"])) for r in d["results"]]


def mcnemar(b, c):
    """双尾精确 McNemar（等价于对不一致对做符号检验）。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


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
    print(f"{'块':<12}{'n':>4}{'A 成功':>13}{'B 成功':>13}"
          f"{'变化':>7}{'换赢/换输':>12}{'p':>8}")

    sizes = [(i * n) // k for i in range(k + 1)]
    block_p = []
    for bi in range(k):
        lo, hi = sizes[bi], sizes[bi + 1]
        chunk = seq[lo:hi]
        if not chunk:
            continue
        ha = sum(1 for _, h in chunk if h)
        hb = sum(1 for t, _ in chunk if b[t])
        win = sum(1 for t, h in chunk if (not h) and b[t])
        loss = sum(1 for t, h in chunk if h and not b[t])
        pv = mcnemar(loss, win)
        block_p.append((bi, win, loss, pv))
        print(f"{f'{lo+1}-{hi}':<12}{len(chunk):>4}"
              f"{f'{ha} ({ha/len(chunk):.0%})':>13}"
              f"{f'{hb} ({hb/len(chunk):.0%})':>13}"
              f"{hb - ha:>+7}{f'+{win}/-{loss}':>12}{pv:>8.3f}")

    win_t = sum(w for _, w, _, _ in block_p)
    loss_t = sum(l for _, _, l, _ in block_p)
    tot_a = sum(1 for _, h in seq if h)
    tot_b = sum(1 for t, _ in seq if b[t])
    p_all = mcnemar(loss_t, win_t)
    print()
    print(f"整体 {tot_a}/{n} = {tot_a/n:.1%} → {tot_b}/{n} = {tot_b/n:.1%}"
          f"（{tot_b - tot_a:+d} | 换赢 {win_t} / 换输 {loss_t}，p = {p_all:.3f}）")
    print()
    print("怎么读：每块 n≈%d，不一致对至少 6:0 才 p<0.05。"
          % (n // k))
    if p_all >= 0.05 and all(pv >= 0.05 for _, _, _, pv in block_p):
        print("→ 统计上看不出差异。这个粒度下分不出遗忘，别把一两笔涨跌当结论。")
    else:
        sig = [(b, w, l, pv) for b, w, l, pv in block_p if pv < 0.05]
        print("→ 有显著块：" + "；".join(
            f"第{b+1}块 换赢{w}/换输{l} 差{l-w:+d} (p={pv:.3f})" for b, w, l, pv in sig))
    return 0


if __name__ == "__main__":
    sys.exit(main())
