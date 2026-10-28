#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""show_fails.py — 失败的任务是怎么停的：停滞 / 打满步数 / 全不安全

用来验证 stall_limit 有没有误伤"每步只挪一点点"的任务——
只要它每步都刷新了历史最好距离，计数就归零，不该被判停滞。

用法：
  python tool/show_fails.py day_result.json
"""

import json
import math
import sys


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "day_result.json"
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    rs = d["results"]
    fails = [r for r in rs if not r["hit"]]
    print(f"{path}：失败 {len(fails)}/{len(rs)}")
    if not fails:
        return 0
    print(f"{'目标':>18} | {'距离':>6} | {'到肩':>5} | {'需θ2':>5} | {'痛':>4} | {'步数':>4} | {'更新':>6} | 停因")
    n_stall = n_max = n_unsafe = 0
    for r in fails:
        steps = r.get("steps", 0)
        dec_up = r.get("dec_up", 0)
        gave = r.get("gave_up", False)
        if steps >= 100:
            why = "打满 max_steps(100)"
            n_max += 1
        elif steps == 0:
            why = "全不安全（一步没动）"
            n_unsafe += 1
        elif gave and steps >= 15:
            why = "停滞 stall_limit(15)"
            n_stall += 1
        elif gave:
            why = f"中途全不安全（第 {steps} 步不敢动）"
            n_unsafe += 1
        else:
            why = "其它"
        tg = r["target"]
        rad = math.hypot(tg[0], tg[1])
        c = max(-1.0, min(1.0, (rad * rad - 2.0) / 2.0))
        need = math.degrees(math.acos(c))
        pain = max(0.0, need - 160.0) * 0.5
        print(f"({tg[0]:>6.2f},{tg[1]:>6.2f}) | {r['dist']:>6.3f} | {rad:>5.2f} | "
              f"{need:>5.0f} | {pain:>4.1f} | {steps:>4} | {dec_up:>6} | {why}")
    print()
    print(f"停滞 {n_stall} | 打满步数 {n_max} | 不安全 {n_unsafe}")
    print("提示：eval_only 不更新决策，dec_up 恒为 0；此时每个状态只评估一次候选动作。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
