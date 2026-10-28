#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2d感觉实验/render_html.py — 把闭环轨迹渲染成 HTML 动画（前端可视化）

读 trajectory.json → 生成 arm_anim.html（Canvas 动画：大臂/小臂/目标/轨迹）

用法：
  python render_html.py --traj trajectory.json --out arm_anim.html
"""

import argparse
import json

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>2D Arm Trajectory</title>
<style>
body { background:#111; color:#eee; font-family:monospace; margin:20px; }
canvas { background:#1a1a1a; border:1px solid #555; }
#info { margin-top:10px; white-space:pre; }
</style></head>
<body>
<h3>2D 手臂闭环（目标：__TARGET__，到达：__HIT__）</h3>
<canvas id="cv" width="640" height="640"></canvas>
<div id="info"></div>
<script>
const DATA = __DATA__;
const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
const W = cv.width, H = cv.height, SCALE = 140, CX = W/2, CY = H/2;
const L1 = 1.0, L2 = 1.0;
let idx = 0;
function toCanvas(x, y) { return [CX + x*SCALE, CY - y*SCALE]; }
function draw() {
  ctx.clearRect(0,0,W,H);
  // 网格/目标
  const [tx, ty] = toCanvas(DATA.target[0], DATA.target[1]);
  ctx.strokeStyle = '#333'; ctx.beginPath();
  for (let i=-2; i<=2; i++) {
    const [gx] = toCanvas(i, 0), [, gy] = toCanvas(0, i);
    ctx.moveTo(gx, 0); ctx.lineTo(gx, H);
    ctx.moveTo(0, gy); ctx.lineTo(W, gy);
  }
  ctx.stroke();
  // 轨迹
  ctx.strokeStyle = '#3a6'; ctx.beginPath();
  DATA.traj.forEach((p,i) => {
    const [px, py] = toCanvas(p.x, p.y);
    if (i===0) ctx.moveTo(px,py); else ctx.lineTo(px,py);
  });
  ctx.stroke();
  // 目标点
  ctx.fillStyle = '#e33'; ctx.beginPath(); ctx.arc(tx,ty,7,0,6.283); ctx.fill();
  // 当前手臂
  const p = DATA.traj[Math.min(idx, DATA.traj.length-1)];
  const a1 = p.t1 * Math.PI/180, a2 = p.t2 * Math.PI/180;
  const ex = L1*Math.cos(a1), ey = L1*Math.sin(a1);
  const hx = ex + L2*Math.cos(a1+a2), hy = ey + L2*Math.sin(a1+a2);
  const [sx,sy] = toCanvas(0,0), [exx,eyy] = toCanvas(ex,ey), [hxx,hyy] = toCanvas(hx,hy);
  ctx.strokeStyle = '#4af'; ctx.lineWidth = 8; ctx.beginPath();
  ctx.moveTo(sx,sy); ctx.lineTo(exx,eyy); ctx.stroke();
  ctx.strokeStyle = '#8cf'; ctx.lineWidth = 6; ctx.beginPath();
  ctx.moveTo(exx,eyy); ctx.lineTo(hxx,hyy); ctx.stroke();
  ctx.fillStyle = '#fff'; ctx.beginPath(); ctx.arc(sx,sy,5,0,6.283); ctx.fill();
  ctx.beginPath(); ctx.arc(hxx,hyy,5,0,6.283); ctx.fill();
  document.getElementById('info').textContent =
    `step ${p.step}/${DATA.traj.length-1} | θ1=${p.t1.toFixed(1)}° θ2=${p.t2.toFixed(1)}° | 手=(${p.x.toFixed(2)},${p.y.toFixed(2)}) | 距离=${p.dist.toFixed(3)}`;
}
function loop() { idx = (idx + 1) % DATA.traj.length; draw(); setTimeout(loop, 120); }
draw(); setTimeout(loop, 600);
</script>
</body>
</html>
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--traj", default="trajectory.json")
    p.add_argument("--out", default="arm_anim.html")
    args = p.parse_args()

    with open(args.traj, encoding="utf-8") as f:
        traj = json.load(f)
    html = (HTML_TEMPLATE
            .replace("__TARGET__", str(traj.get("target")))
            .replace("__HIT__", str(traj.get("hit")))
            .replace("__DATA__", json.dumps(traj, ensure_ascii=False)))
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[html] {args.out}（浏览器打开看动画）", flush=True)


if __name__ == "__main__":
    main()
