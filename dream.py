#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dream.py — 规则世界里的训练（**预训练** 与 **夜间复习** 共用同一套）

为什么抽出来：
  夜间"复习"在旧版里是另一套数据（随机目标、没有近目标、世界动作全是 ±60），
  名义上防遗忘，实际是在用不同分布覆盖 base——实测把"目标附近 0.1 米的微调"
  洗掉了，睡后完成度从 50% 掉到 22%。
  复习 = 重温同一套数据 + 同一套损失，所以数据和损失都放在这里，两边只能一样。

数据分布：
  20% 目标就在手上（学"到了别乱动"）
  10% 目标在 0.05~0.2 米内（学小动作微调）
  70% 随机可达目标
  世界监督动作：30% ±10°（精细），70% ±60°（大动作）
"""

import math

import env


def sample(rng):
    """采一条训练样本：返回 (t1, t2, target, d1r, d2r)；目标不可达返回 None。

    d1r/d2r 是给世界模型做监督用的动作（度），与决策动作无关。
    """
    t1 = rng.uniform(-180.0, 180.0)
    t2 = rng.uniform(env.ELBOW_MIN, env.FOLD_MAX)   # 160~180 是硬挤进去的，也真能到
    r0 = rng.random()
    if r0 < 0.2:                                    # 目标就在手上
        hx0, hy0 = env.forward_kinematics(t1, t2)
        target = (hx0, hy0)
    elif r0 < 0.3:                                  # 近目标（学小动作）
        hx0, hy0 = env.forward_kinematics(t1, t2)
        ang = rng.uniform(0.0, 6.28)
        r = rng.uniform(0.05, 0.2)
        target = (hx0 + r * math.cos(ang), hy0 + r * math.sin(ang))
    else:                                           # 随机可达目标
        target = (rng.uniform(-1.8, 1.8), rng.uniform(-1.8, 1.8))
    if env.inverse_kinematics(*target) is None:
        return None
    if rng.random() < 0.3:                          # 30% 精细动作，让世界学准小动作
        d1r, d2r = rng.uniform(-10.0, 10.0), rng.uniform(-10.0, 10.0)
    else:
        d1r, d2r = rng.uniform(-60.0, 60.0), rng.uniform(-60.0, 60.0)
    return t1, t2, target, d1r, d2r


def decision_loss(dec, obs, t1, t2, target):
    """决策：规则世界想象（位置按阻力后的角度，痛按结果角度）。返回 (loss, v)。"""
    import torch
    d1, d2 = dec(obs)
    nt2 = env.step_theta2_t(t2, d2)
    a1 = (t1 + d1) * math.pi / 180.0
    a2 = a1 + nt2 * math.pi / 180.0
    hx = torch.cos(a1) + torch.cos(a2)
    hy = torch.sin(a1) + torch.sin(a2)
    dist = torch.sqrt((hx - target[0]) ** 2 + (hy - target[1]) ** 2 + 1e-6)
    pain = env.pain_t(nt2)
    v = -dist - 5.0 * (torch.exp(pain / 10.0) - 1.0)
    return -v.mean(), v


def world_loss(world, obs, act, t1, t2, d1r, d2r, scale):
    """世界：规则数据监督（位置相对加权 + 痛项指数差）。"""
    import torch
    pred = world(obs, act)
    nt1, nt2, _want2, pain_a, load_a = env.step_body(t1, t2, d1r, d2r)
    nhx, nhy = env.forward_kinematics(nt1, nt2)
    ohx, ohy = env.forward_kinematics(t1, t2)
    actual = torch.tensor([nhx - ohx, nhy - ohy, 0.0, pain_a, load_a],
                          device=scale.device) / scale
    diff = pred.squeeze(0) - actual
    w_pos = 1.0 / (actual[:2].abs() + 0.05) ** 2      # 小动作权重大
    loss = (diff[:2] ** 2 * w_pos).mean() + (diff[2:] ** 2).mean()
    return loss + env.pain_loss_t(pred[0, 3] * scale[3], actual[3] * scale[3])
