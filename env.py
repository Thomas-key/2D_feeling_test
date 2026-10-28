#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2d感觉实验/env.py — 2D 两关节手臂环境（纯数学，不建 3D）

右手：肩 θ1（0~150°）、肘 θ2（0~150°）
正运动学：手位置 = 大臂 + 小臂（2D 平面）
感觉（数字）：
  - 本体：θ1, θ2（角度）、角速度
  - 触觉：手端到球的距离（接触时升高）
  - 痛：想推到的角度超出身体范围的量（越推越痛；痛不累积，按动作算）
  - 负载：目标距离（远 → 负载高）
动作：关节增量（档位，如 -15~+15 度/步，7 档）
"""

import math

import numpy as np

L1 = 1.0      # 大臂长度
L2 = 1.0      # 小臂长度
TH_MIN, TH_MAX = 0.0, 150.0   # 旧的安全范围（保留兼容）
ELBOW_MIN, ELBOW_MAX = 0.0, 180.0   # 肘关节物理范围（0=折叠，180=伸直）
SHOULDER_RANGE = None               # 肩无限制
STEP_DEG = 5.0                # 每档增量（度）
N_BINS = 7                    # 动作档位：-3..+3（×5°）


def forward_kinematics(theta1, theta2):
    """角度（度）→ 手位置（x, y）；y 向上。"""
    a1 = math.radians(theta1)
    a2 = math.radians(theta2)
    x = L1 * math.cos(a1) + L2 * math.cos(a1 + a2)
    y = L1 * math.sin(a1) + L2 * math.sin(a1 + a2)
    return x, y


def inverse_kinematics(x, y):
    """目标位置 → 关节角度（度，0~150）；不可达返回 None。"""
    d = (x * x + y * y - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    if d < -1.0 or d > 1.0:
        return None
    a2 = math.degrees(math.acos(d))                 # 肘（0~180）
    a1 = math.degrees(math.atan2(y, x) - math.atan2(L2 * math.sin(math.radians(a2)),
                                                    L1 + L2 * math.cos(math.radians(a2))))
    a1 = (a1 + 180) % 360 - 180      # 归一化到 [-180, 180]（肩无限制）
    if not (ELBOW_MIN <= a2 <= ELBOW_MAX):
        return None
    return a1, a2


FOLD_PAIN = 160.0   # θ2 > 160 → 折叠痛（挤压）
# 世界模型输出尺度（归一化/反归一化）：delta_x, delta_y, touch, pain, load
OUT_SCALE = [2.0, 2.0, 1.0, 30.0, 1.0]
FOLD_LIMIT = 160.0  # θ2 折叠极限（动作超此 → 被挡——角度不变+负载升）


def pain(theta1, theta2):
    """痛：肘角超出物理范围 [0,180] + 折叠（>160）挤压痛。"""
    p = 0.0
    if theta2 < ELBOW_MIN:
        p += ELBOW_MIN - theta2
    elif theta2 > ELBOW_MAX:
        p += theta2 - ELBOW_MAX
    if theta2 > FOLD_PAIN:
        p += (theta2 - FOLD_PAIN) * 0.5     # 折叠挤压（160→180 逐渐痛）
    return p


def self_touch(theta1, theta2):
    """自触觉：小臂贴到大臂（θ2 接近 0，折叠）→ 接触升高。"""
    if theta2 <= 0:
        return 1.0
    if theta2 >= 15:
        return 0.0
    return 1.0 - theta2 / 15.0


def touch(theta1, theta2, ball):
    """触觉：手端到球的距离近 → 触觉高（0~1）。"""
    if ball is None:
        return 0.0
    hx, hy = forward_kinematics(theta1, theta2)
    dist = math.hypot(hx - ball[0], hy - ball[1])
    r = ball[2] if len(ball) > 2 else 0.2
    return max(0.0, 1.0 - max(0.0, dist - r) / 0.8)


def load(theta1, theta2, ball=None):
    """肌肉负载：碰球（受阻）+ 折叠（θ2>160 挤压）时高。"""
    ld = 0.0
    if theta2 > FOLD_PAIN:
        ld = min(1.0, (theta2 - FOLD_PAIN) / 20.0)   # 折叠负载 0→1
    if ball is not None:
        ld = max(ld, touch(theta1, theta2, ball))
    return ld


PAIN_CAP = 30.0     # 痛觉上限（生物上会饱和；也保证归一化后落在 [0,1]）


def step_body(theta1, theta2, d1, d2):
    """按动作推进一步：位置被身体挡住，感觉按"想去的角度"算。

    位置：θ2 最多到 FOLD_LIMIT（160），再推也不动。
    感觉：痛/负载由"想推到的角度"决定——顶住和痛来自发力，不来自到达。
    痛不累积：每一步只看这一步推多狠，松手就不痛。

    返回 (θ1', θ2', want2, pain, load)
    """
    want2 = theta2 + d2
    nt1 = theta1 + d1
    nt2 = min(FOLD_LIMIT, max(ELBOW_MIN, want2))
    return nt1, nt2, want2, min(PAIN_CAP, pain(nt1, want2)), load(nt1, want2)


def pain_t(want2):
    """step_body 的可微版痛（与 pain() 同一公式 + PAIN_CAP 饱和）。

    决策价值靠它把梯度传回动作——"预测掰到 180 会很痛"就是拿超限动作问它。
    """
    import torch
    p = torch.relu(-want2) + torch.relu(want2 - ELBOW_MAX)
    p = p + torch.relu(want2 - FOLD_PAIN) * 0.5
    return torch.clamp(p, max=PAIN_CAP)


def feeling_text(theta1, theta2, vel, ball, target):
    """感觉文本化（Jev 式输入）。"""
    t = max(touch(theta1, theta2, ball), self_touch(theta1, theta2))
    p = pain(theta1, theta2)
    ld = max(load(theta1, theta2, ball), self_touch(theta1, theta2))
    return (f"角度: {theta1:.0f} {theta2:.0f} | 速度: {vel:.1f} | "
            f"触觉: {t:.2f} | 痛: {p:.0f} | 负载: {ld:.2f}")


def best_action(theta1, theta2, target, ball=None, safe=True):
    """规则正解：朝目标；痛高 → 反向+减幅（保守，远离极限）。"""
    if ball is not None and target is None:
        target = (ball[0], ball[1])
    if target is None:
        return 0, 0
    ik = inverse_kinematics(target[0], target[1])
    if ik is None:
        hx, hy = forward_kinematics(theta1, theta2)
        ang = math.degrees(math.atan2(target[1] - hy, target[0] - hx))
        a1 = (ang + 360) % 360
        ik = (min(TH_MAX, max(TH_MIN, a1)), theta2)
    t1, t2 = ik
    d1 = (t1 - theta1) / STEP_DEG
    d2 = (t2 - theta2) / STEP_DEG
    d1 = int(np.clip(round(d1), -3, 3))
    d2 = int(np.clip(round(d2), -3, 3))
    if safe:
        # 肘规则：当前已超范围 / 动作会超范围 → 反向 + 减幅
        over_now = (theta2 < ELBOW_MIN) or (theta2 > ELBOW_MAX)
        nxt = theta2 + d2 * STEP_DEG
        over_next = (nxt < ELBOW_MIN) or (nxt > ELBOW_MAX)
        if over_now or over_next:
            center = (ELBOW_MIN + ELBOW_MAX) / 2
            sign = 1 if theta2 < center else -1
            d2 = sign * max(1, min(2, abs(d2) if d2 != 0 else 2))
            d2 = int(np.clip(d2, -3, 3))
    return d1, d2


# ===== 分层档位（3 级 × 5 档，去重后 15 个具体角度）=====
GEAR_ANGLES = [-180, -90, -45, -30, -20, -15, -8, 0, 8, 15, 20, 30, 45, 90, 180]


def gear_index(angle):
    """角度增量 → 最近的档位索引（0..14）。"""
    import bisect
    return min(range(len(GEAR_ANGLES)), key=lambda i: abs(GEAR_ANGLES[i] - angle))


def gear_angle(idx):
    return GEAR_ANGLES[idx]


# ===== 网格感觉（16 网格：大臂 4 段×2 侧 + 小臂 4 段×2 侧，每网格触觉/负载/痛）=====
N_GRID = 16


def arm_grid_points(theta1, theta2):
    """16 网格点世界坐标：大臂 8（4 段×2 侧）+ 小臂 8。"""
    a1 = math.radians(theta1)
    a2 = math.radians(theta2)
    pts = []
    for side in (-1, 1):          # 里/外
        for i in range(4):        # 大臂 4 段
            t = (i + 0.5) / 4.0
            ex = L1 * math.cos(a1)
            ey = L1 * math.sin(a1)
            # 侧向：(垂直于大臂方向)
            nx, ny = -math.sin(a1), math.cos(a1)
            pts.append((ex * t + side * 0.08 * nx, ey * t + side * 0.08 * ny))
    for side in (-1, 1):
        for i in range(4):        # 小臂 4 段
            t = (i + 0.5) / 4.0
            ex = L1 * math.cos(a1)
            ey = L1 * math.sin(a1)
            fx = ex + L2 * math.cos(a1 + a2) * t
            fy = ey + L2 * math.sin(a1 + a2) * t
            nx, ny = -math.sin(a1 + a2), math.cos(a1 + a2)
            pts.append((fx + side * 0.08 * nx, fy + side * 0.08 * ny))
    return pts


def grid_sensation(theta1, theta2, ball=None):
    """48 维感觉向量：[触觉×16, 负载×16, 痛×16]。"""
    pts = arm_grid_points(theta1, theta2)
    touch = [0.0] * N_GRID
    loadv = [0.0] * N_GRID
    painv = [0.0] * N_GRID
    p = pain(theta1, theta2) / 30.0
    # 折叠：θ2 > 150（小臂贴大臂——反向折叠）→ 触觉/负载
    if theta2 > 150:
        fold = min(1.0, (theta2 - 150.0) / 10.0)
        for i in range(8, 16):
            touch[i] = fold
            loadv[i] = fold
    # 碰球：网格点接近球
    if ball is not None:
        r = ball[2] if len(ball) > 2 else 0.2
        for i, (px, py) in enumerate(pts):
            d = math.hypot(px - ball[0], py - ball[1])
            if d < r + 0.1:
                touch[i] = max(touch[i], 1.0 - d / (r + 0.1))
                loadv[i] = max(loadv[i], touch[i])
    # 痛：小臂网格
    if p > 0:
        for i in range(8, 16):
            painv[i] = min(1.0, p)
    return touch + loadv + painv   # 48 维


# ===== 数值观测向量（纯数值，归一化去量纲）=====
def obs_vector(t1, t2, target):
    """数值观测：[角度/180, 手位置/2, 目标/2, 距离/2, 方向/180] + 感觉48 → 56 维。"""
    hx, hy = forward_kinematics(t1, t2)
    dist = math.hypot(hx - target[0], hy - target[1])
    ang = math.degrees(math.atan2(target[1] - hy, target[0] - hx))
    # 角度用 sin/cos 编码（连续——无 ±180 边界跳变）
    a1 = math.radians(t1)
    a2 = math.radians(t2)
    return ([math.sin(a1), math.cos(a1), math.sin(a2), math.cos(a2),
             hx / 2.0, hy / 2.0,
             target[0] / 2.0, target[1] / 2.0, dist / 2.0, math.sin(math.radians(ang)),
             math.cos(math.radians(ang))]
            + grid_sensation(t1, t2, None))
