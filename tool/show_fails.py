#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""show_fails.py — 失败的任务是怎么停的：停滞 / 打满步数 / 全不安全

用来验证 stall_limit 有没有误伤"每步只挪一点点"的任务——
只要它每步都刷新了历史最好距离，计数就归零，不该被判停滞。

用法：
  python tool/show_fails.py day_result.json            # 逐个任务
  python tool/show_fails.py day_result.json --summary  # 只看最终统计（批量对比用）
"""

import argparse
import math


def why_of(r):
    steps = r.get("steps", 0)
    gave = r.get("gave_up", False)
    if steps >= 100:
        return "max", "打满 max_steps(100)"
    if steps == 0:
        return "unsafe", "全不安全（一步没动）"
    if gave and steps >= 15:
        return "stall", "停滞 stall_limit(15)"
    if gave:
        return "unsafe", f"中途全不安全（第 {steps} 步不敢动）"
    return "other", "其它"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path", nargs="?", default="day_result.json")
    p.add_argument("--summary", action="store_true", help="只打印最后的统计")
    args = p.parse_args()

    import json
    with open(args.path, encoding="utf-8") as f:
        d = json.load(f)
    rs = d["results"]
    fails = [r for r in rs if not r["hit"]]
    n_stall = n_max = n_unsafe = n_other = 0
    rows = []
    for r in fails:
        kind, why = why_of(r)
        if kind == "stall":
            n_stall += 1
        elif kind == "max":
            n_max += 1
        elif kind == "unsafe":
            n_unsafe += 1
        else:
            n_other += 1
        rows.append((r, why))

    if not args.summary:
        print(f"{args.path}：失败 {len(fails)}/{len(rs)}")
        if fails:
            print(f"{'目标':>18} | {'距离':>6} | {'到肩':>5} | {'需θ2':>5} | {'痛':>4} | "
                  f"{'步数':>4} | {'更新':>6} | 停因")
            for r, why in rows:
                tg = r["target"]
                rad = math.hypot(tg[0], tg[1])
                c = max(-1.0, min(1.0, (rad * rad - 2.0) / 2.0))
                need = math.degrees(math.acos(c))
                pain = max(0.0, need - 160.0) * 0.5
                print(f"({tg[0]:>6.2f},{tg[1]:>6.2f}) | {r['dist']:>6.3f} | {rad:>5.2f} | "
                      f"{need:>5.0f} | {pain:>4.1f} | {r.get('steps', 0):>4} | "
                      f"{r.get('dec_up', 0):>6} | {why}")
            print()

    print(f"{args.path}：失败 {len(fails)}/{len(rs)} | "
          f"停滞 {n_stall} | 打满步数 {n_max} | 不安全 {n_unsafe}"
          f"{f' | 其它 {n_other}' if n_other else ''}")
    if d.get("eval_only"):
        print("提示：这是 --eval-only 的结果，dec_up 恒为 0；每个状态只评估一次候选动作。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
