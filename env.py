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
FOLD_LIMIT = 160.0  # 折叠阻力点：之后每推 1° 只挤进去 FOLD_YIELD°
FOLD_MAX = 180.0    # 绝对上限：到这儿再推也不动
FOLD_YIELD = 0.25   # 阻力系数（0.25 = 要费约 4 倍的劲）

_DEFAULT_BODY = {
    "d1_bias": 0.0,        # 肩每步实际多转这么多度（校准偏差）
    "d2_bias": 0.0,        # 肘每步实际多转这么多度
    "fold_start": FOLD_LIMIT,   # 折叠阻力从几度开始
    "fold_yield": FOLD_YIELD,   # 阻力系数
    "pain_start": FOLD_PAIN,    # 痛从几度开始
}
# 可变的"身体"：默认等于上面的常量。`--body k=v` 可在运行时改。
# 目的：让"世界变了"成为一个**可注入的实验变量**，而不是改代码。
# 注意：预训练永远用默认值——变了之后才能测"能不能调回来"。
BODY = dict(_DEFAULT_BODY)


def set_body(**kw):
    """运行时改动身体。未知的键直接报错（别让拼错的参数静悄悄无效）。"""
    for k, v in kw.items():
        if k not in BODY:
            raise KeyError(f"未知的身体参数 {k}；可选：{sorted(BODY)}")
        BODY[k] = float(v)
    return dict(BODY)


def body_change():
    """和默认值不同的项（全默认返回空 dict）。"""
    return {k: v for k, v in BODY.items() if abs(v - _DEFAULT_BODY[k]) > 1e-12}


def parse_body_args(pairs):
    """把 ["d2_bias=3"] 解析成 {"d2_bias": 3.0} 并应用到 BODY。"""
    out = {}
    for s in pairs or []:
        if "=" not in s:
            raise SystemExit(f"--body 要写成 key=value，收到 {s!r}")
        k, v = s.split("=", 1)
        out[k.strip()] = float(v)
    if out:
        set_body(**out)
    return out


def _act(d1, d2):
    """把"命令的动作"换成"身体实际做的动作"（校准偏差在这里注入）。"""
    return d1 + BODY["d1_bias"], d2 + BODY["d2_bias"]


def pain(theta1, theta2):
    """痛：肘角超出物理范围 [0,180] + 折叠（>160）挤压痛。"""
    p = 0.0
    if theta2 < ELBOW_MIN:
        p += ELBOW_MIN - theta2
    elif theta2 > ELBOW_MAX:
        p += theta2 - ELBOW_MAX
    if theta2 > BODY["pain_start"]:
        p += (theta2 - BODY["pain_start"]) * 0.5   # 折叠挤压（越挤越痛）
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
    if theta2 > BODY["pain_start"]:
        ld = min(1.0, (theta2 - BODY["pain_start"]) / 20.0)   # 折叠负载 0→1
    if ball is not None:
        ld = max(ld, touch(theta1, theta2, ball))
    return ld


PAIN_CAP = 30.0     # 痛觉上限（生物上会饱和；也保证归一化后落在 [0,1]）


def fold_target(theta2, d2):
    """把"想推到的角度"折算成身体实际能到的角度。

    160° 之前自由活动；之后是**阻力不是墙**——每推 1° 只挤进去 FOLD_YIELD°，
    一直可以挤到 FOLD_MAX。所以推得越狠折得越深，也就越痛（且能一直更痛）。
    0° 一侧仍是硬夹。
    """
    fold, yield_ = BODY["fold_start"], BODY["fold_yield"]
    want2 = theta2 + d2
    if want2 <= fold:
        return max(ELBOW_MIN, want2)                     # 自由区
    if theta2 >= fold:                                   # 已在阻力区：整个动作打折
        nt2 = theta2 + d2 * yield_
    else:                                                # 从自由区越界：只有超出部分打折
        nt2 = fold + (want2 - fold) * yield_
    return min(FOLD_MAX, nt2)


def step_body(theta1, theta2, d1, d2):
    """按动作推进一步：位置受阻力限制，感觉由**结果状态**决定。

    返回 (θ1', θ2', want2, pain, load)
      want2 只是"想推到的角度"（诊断用）；痛/负载都按实际到达的 θ2' 算。
    """
    d1, d2 = _act(d1, d2)          # 身体不是完全听命令的（校准偏差）
    nt1 = theta1 + d1
    nt2 = fold_target(theta2, d2)
    return nt1, nt2, theta2 + d2, min(PAIN_CAP, pain(nt1, nt2)), load(nt1, nt2)


def step_theta2_t(theta2, d2):
    """fold_target 的可微版（决策价值里"想象"用），分支与它逐条对应。"""
    import torch
    d2 = d2 + BODY["d2_bias"]                 # 与 step_body 里的 _act 一致
    fold, yield_ = BODY["fold_start"], BODY["fold_yield"]
    want2 = theta2 + d2
    if theta2 >= fold:
        nt2 = torch.where(want2 <= fold, want2, theta2 + d2 * yield_)
    else:
        nt2 = torch.where(want2 <= fold, want2, fold + (want2 - fold) * yield_)
    return torch.clamp(nt2, min=ELBOW_MIN, max=FOLD_MAX)


def pain_t(theta2):
    """可微版痛（与 pain() 同一公式 + PAIN_CAP 饱和），吃**实际到达**的角度。"""
    import torch
    p = torch.relu(-theta2) + torch.relu(theta2 - ELBOW_MAX)
    p = p + torch.relu(theta2 - BODY["pain_start"]) * 0.5
    return torch.clamp(p, max=PAIN_CAP)


def pain_loss_t(pred_pain, true_pain):
    """痛项损失：价值空间**相对**加权（和位置项的 1/(|Δ|+0.05)² 同构）。

    为什么不能用 0.5·(e^p − e^a)²：那个按痛的**大小**加权，于是
      "该是 0 却报 0.7" 的损失只有 0.0026（价值上却错了 0.36），
      "该是 8 却报 7" 的损失有 0.022（价值上错 5）——模型被允许在无痛区乱报，
    而那正是"折了一下到底痛不痛"的判定边界。
    相对加权后：前者 0.131（×50），后者 0.022（×1）——无痛区终于开始算数。
    """
    import torch
    pen_p = 5.0 * (torch.exp(torch.clamp(pred_pain, -30.0, 60.0) / 10.0) - 1.0)
    pen_a = 5.0 * (torch.exp(torch.clamp(true_pain, -30.0, 60.0) / 10.0) - 1.0)
    return (pen_p - pen_a) ** 2 / (pen_a.abs() + 1.0) ** 2


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
    # 折叠：越过阻力点（160）小臂才真正被压住 → 触觉/负载（与痛边界一致）
    if theta2 > BODY["fold_start"]:
        fold = min(1.0, (theta2 - BODY["fold_start"]) / 20.0)
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
    """数值观测（61 维）：
       sin/cos(θ1), θ1/180（先环绕到 [-180,180)）, sin/cos(θ2), θ2/180,
       手位置/2, 目标/2, 距离/4, 方向 sin/cos, 感觉网格 48

    角度给两套编码：sin/cos 让位置这类三角量好算且无 ±180 跳变；
    线性值让"阈值/阻力"这类规则（θ2>160 才痛）在网络里是一个平面，不用反解三角函数。
    肩可无限旋转，所以线性项必须先环绕——否则 θ1=229° 会喂出 1.27（训练分布之外）。
    """
    hx, hy = forward_kinematics(t1, t2)
    dist = math.hypot(hx - target[0], hy - target[1])
    ang = math.degrees(math.atan2(target[1] - hy, target[0] - hx))
    a1 = math.radians(t1)
    a2 = math.radians(t2)
    t1w = (t1 + 180.0) % 360.0 - 180.0    # 肩可无限旋转 → 先环绕到 [-180,180)
    return ([math.sin(a1), math.cos(a1), t1w / 180.0,
             math.sin(a2), math.cos(a2), t2 / 180.0,
             hx / 2.0, hy / 2.0,
             target[0] / 2.0, target[1] / 2.0,
             dist / 4.0,
             math.sin(math.radians(ang)), math.cos(math.radians(ang))]
            + grid_sensation(t1, t2, None))
