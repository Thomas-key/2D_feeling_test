#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_tasks.py — 生成固定的任务集（存档），以后所有对比都跑同一个文件

为什么需要：
  以前每次 `--tasks N` 都是现场随机抽的，同一个命令两次能得到不同的任务集
  （截图里的 ±2 个任务波动有一半来自这里）。没有固定任务集，
  "提升了 3 个点" 和噪声根本分不开。

  `--tasks N` 其实是"抽 N 次"，不可达的会丢掉 —— 所以 N 不是任务数。
  历史记录：`--seed 1 --draws 100` 正好得到 87 个任务，就是 README 里 88.5% 那批。

用法：
  python tool/make_tasks.py --seed 1 --draws 100 --out tasks/seed1.json
  python tool/make_tasks.py --seed 2 --draws 100 --out tasks/seed2.json --exclude tasks/seed1.json

--exclude 用来做"两个数据集不重叠"：新任务如果离已有任务太近（--min-sep）就丢掉。
  默认 0 = 不做分离（保持和历史抽取完全一致）。
"""

import argparse
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import env  # noqa: E402

PAIN_RADIUS = 0.347   # 到肩小于这个距离必须折过 160°（见 DESIGN 八）


def load_tasks(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return [tuple(t) for t in (d["tasks"] if isinstance(d, dict) else d)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1, help="抽取用的随机种子")
    p.add_argument("--draws", type=int, default=100, help="抽多少次（不是任务数）")
    p.add_argument("--out", required=True, help="输出文件（tasks/seed1.json）")
    p.add_argument("--exclude", nargs="*", default=[],
                   help="已有任务集：新任务离它们太近就丢掉（做不重叠的两批）")
    p.add_argument("--min-sep", type=float, default=0.0,
                   help="与 exclude 里任务的最小距离（米）；0 = 不检查")
    args = p.parse_args()

    others = []
    for path in args.exclude:
        others += load_tasks(path)
    if args.min_sep > 0 and not others:
        print("[warn] --min-sep 给了但没有 --exclude，等于不检查", flush=True)

    rng = random.Random(args.seed)
    tasks, unreachable, too_close = [], 0, 0
    for _ in range(args.draws):
        tg = (rng.uniform(-1.8, 1.8), rng.uniform(-1.8, 1.8))
        if env.inverse_kinematics(*tg) is None:
            unreachable += 1
            continue
        if args.min_sep > 0:
            near = min((math.hypot(tg[0] - o[0], tg[1] - o[1]) for o in others), default=9.9)
            if near < args.min_sep:
                too_close += 1
                continue
        tasks.append([round(tg[0], 6), round(tg[1], 6)])

    if not tasks:
        print("[error] 一个任务都没抽到", flush=True)
        return 2

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "draws": args.draws, "count": len(tasks),
                   "exclude": args.exclude, "min_sep": args.min_sep,
                   "tasks": tasks}, f, ensure_ascii=False)

    rs = [math.hypot(*t) for t in tasks]
    pain = sum(1 for r in rs if r < PAIN_RADIUS)
    edge = sum(1 for r in rs if 0.32 <= r <= 0.40)
    print(f"[save] {args.out}")
    print(f"  抽取 {args.draws} 次 → 不可达丢掉 {unreachable}"
          f"{f'，太近丢掉 {too_close}' if args.min_sep > 0 else ''}"
          f" → **{len(tasks)} 个任务**")
    print(f"  到肩距离 r：最小 {min(rs):.3f} / 中位 {sorted(rs)[len(rs) // 2]:.3f} / 最大 {max(rs):.3f}")
    print(f"  需要折进痛区（r < {PAIN_RADIUS}）{pain} 个 | 折叠边缘（0.32~0.40）{edge} 个")
    if args.exclude:
        dmin = min(math.hypot(a[0] - b[0], a[1] - b[1]) for a in tasks for b in others)
        print(f"  与 exclude 里任务的最近距离 {dmin:.3f} 米")
    return 0


if __name__ == "__main__":
    sys.exit(main())
