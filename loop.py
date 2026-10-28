#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""loop.py — 决策 / 世界共用的任务循环（train_day.py 与 closed_loop_exec.py 都调这一个）

为什么要有这个文件：
  这段循环原先在 train_day.py 和 closed_loop_exec.py 里各抄了一份，而且抄歪了——
  closed_loop 有内层重试（一个状态上最多试 300 次），train_day 没有（拒绝一次就烧掉一格 step）。
  于是两边的 step 不是同一个时钟，对比轨迹时得出了一个假 bug（"权重一模一样结果却不同"）。
  现在只有这一份。

步的定义（两边一致）：
  step  = 一次"被接受的移动"
  retry = 同一个状态上最多试几次（拒绝 → 更新决策 → 原地再试，不占 step 预算）

价值：
  V_new = -dist' - 5·(exp(pain'/10) - 1) - 2·load'
  验证：V_new > V_cur + ACCEPT_EPS 才执行（真实价值 V_cur 用环境真值，不用世界模型）
"""

import math

import torch

import env

HIT_DIST = 0.1      # 到达判定（米）
ACCEPT_EPS = 0.01   # 价值至少提升这么多才执行（平手不走）
MAX_PAIN = 30.0     # v_new 里的痛上限（防 exp 爆）


def obs_tensor(t1, t2, target, device):
    return torch.tensor(env.obs_vector(t1, t2, target), dtype=torch.float32,
                        device=device).unsqueeze(0)


def true_value(t1, t2, target):
    """真实价值（环境真值，不是世界模型预测）。"""
    hx, hy = env.forward_kinematics(t1, t2)
    dist = math.hypot(hx - target[0], hy - target[1])
    p = env.pain(t1, t2)
    return -dist - 5.0 * (math.exp(min(p, MAX_PAIN) / 10.0) - 1.0) - 2.0 * env.load(t1, t2, None)


def vstar(target):
    """V*：可达 → 0；不可达 → -(d-2)（最近可达点）。"""
    d = math.hypot(target[0], target[1])
    return 0.0 if d <= 2.0 else -(d - 2.0)


def state_row(step, t1, t2, target):
    """轨迹里的一帧（给 view.html 画）。"""
    hx, hy = env.forward_kinematics(t1, t2)
    return {"step": step,
            "t1": round((t1 + 180.0) % 360.0 - 180.0, 1), "t2": round(t2, 1),
            "x": round(hx, 2), "y": round(hy, 2),
            "dist": round(math.hypot(hx - target[0], hy - target[1]), 2)}


def _world_update(world, opt_w, obs, act, pred, o1, o2, t1, t2, scale):
    """执行后用真实结果监督世界模型（门控：只学预测得离谱的地方）。返回是否更新了。"""
    nhx, nhy = env.forward_kinematics(t1, t2)
    ohx, ohy = env.forward_kinematics(o1, o2)
    raw = torch.tensor([nhx - ohx, nhy - ohy, 0.0,
                        env.pain(t1, t2), env.load(t1, t2, None)], device=scale.device)
    actual = raw / scale
    err = (pred.squeeze(0) - actual).abs()
    if float(err[:2].max()) <= 0.1 and float(err[3]) <= 0.1:
        return False
    pred2 = world(obs, act.detach())
    loss = ((pred2.squeeze(0) - actual) ** 2).mean()
    # 痛项用指数差（与价值一致——不是线性 MSE）
    pp = torch.exp(torch.clamp(pred2[0, 3] * scale[3], -30.0, 60.0) / 10.0)
    pa = torch.exp(torch.clamp(actual[3] * scale[3], -30.0, 60.0) / 10.0)
    loss = loss + 0.5 * (pp - pa) ** 2
    opt_w.zero_grad()
    loss.backward()
    opt_w.step()
    return True


def load_base(dec, world, path, dec_lora=False, world_lora=True):
    """加载 base（pretrain_base.pt / night_base.pt）。

    dec_lora / world_lora：检查点里带着 LoRA 时，要不要一起加载。
      世界默认加载——睡一觉后对世界的适应算 base 的一部分（老 train_day 就是这样）。
      决策默认不加载——白天从"干净的 base"出发（老 train_day 也没加载）。
    """
    ckpt = torch.load(path, map_location="cpu")
    dec.base.load_state_dict(ckpt["dec_base"])
    for k in ("head_shoulder", "head_elbow"):
        getattr(dec, k).load_state_dict(ckpt["dec_heads"][k])
    world.base.load_state_dict(ckpt["world_base"])
    world.heads.load_state_dict(ckpt["world_heads"])
    if dec_lora and "dec_lora" in ckpt:
        for k, lora in dec.loras().items():
            lora.load_state_dict(ckpt["dec_lora"][k])
    if world_lora and "world_lora" in ckpt:
        for k, lora in world.loras_dict().items():
            lora.load_state_dict(ckpt["world_lora"][k])


def run_task(dec, world, opt_d, opt_w, target, t1, t2, *,
             max_steps=100, max_retries=300, eval_only=False, verbose=False):
    """跑一个任务（从 (t1,t2) 出发）。

    eval_only=True：权重全冻结——拒绝不更新、直接放弃这一步（只是评估策略，不学习）。

    返回：
      hit      是否到达（dist < HIT_DIST）
      traj     经过的每一帧（含起点和终点）
      dec_up   决策更新次数
      wm_up    世界更新次数
      t1,t2    最终关节角（度）
      dist     最终距离
      steps    被接受的移动次数
      gave_up  是否因为"一个状态试满 max_retries 次还不通过"而放弃
    """
    device = next(dec.parameters()).device
    scale = torch.tensor(env.OUT_SCALE, device=device)
    traj = [state_row(0, t1, t2, target)]
    dec_up = wm_up = 0
    steps = 0
    hit = gave_up = False
    dist = traj[0]["dist"]
    v_new = v_cur = 0.0

    for step in range(max_steps):
        hx, hy = env.forward_kinematics(t1, t2)
        dist = math.hypot(hx - target[0], hy - target[1])
        if dist < HIT_DIST:
            hit = True
            break
        v_cur = true_value(t1, t2, target)
        obs = obs_tensor(t1, t2, target, device)   # 重试期间状态不变 → 只算一次
        ok = False
        tries = 0
        for retry in range(max_retries):
            tries = retry + 1
            d1, d2 = dec(obs)
            act = torch.cat([d1, d2], dim=-1) / 60.0
            pred = world(obs, act)
            pn = pred * scale                                   # 反归一化
            nhx, nhy = hx + pn[0, 0], hy + pn[0, 1]
            dist_t = torch.sqrt((nhx - target[0]) ** 2 + (nhy - target[1]) ** 2 + 1e-6)
            pain_t = torch.clamp(pn[0, 3], 0.0, MAX_PAIN)
            v_new = -dist_t - 5.0 * (torch.exp(pain_t / 10.0) - 1.0) - 2.0 * pn[0, 4]

            if float(v_new) > v_cur + ACCEPT_EPS:
                o1, o2 = t1, t2
                t1 = t1 + float(d1)
                t2 = min(env.FOLD_LIMIT, max(env.ELBOW_MIN, t2 + float(d2)))  # 折叠 160 挡
                if not eval_only:
                    if _world_update(world, opt_w, obs, act, pred, o1, o2, t1, t2, scale):
                        wm_up += 1
                ok = True
                if verbose:
                    print(f"[step {step}] 重试 {retry} | v_new {float(v_new):.3f} | "
                          f"v_cur {v_cur:.3f} | dist {dist:.2f}", flush=True)
                break

            if eval_only:          # 冻结时不更新 → 原地再试也没意义
                break

            loss_d = -v_new        # 拒绝 → 决策往上抬
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()
            dec_up += 1

        if not ok:
            gave_up = True
            if verbose:
                print(f"[step {step}] 重试 {tries} 次仍不通过 → 放弃 | "
                      f"v_new {float(v_new):.3f} | v_cur {v_cur:.3f} | "
                      f"dist {dist:.2f}", flush=True)
            break

        steps = step + 1
        traj.append(state_row(steps, t1, t2, target))

    # 最终距离按现在的 (t1,t2) 重算（循环跑满时 dist 还停在最后一次移动之前）
    hx, hy = env.forward_kinematics(t1, t2)
    dist = math.hypot(hx - target[0], hy - target[1])
    return {"target": tuple(target), "hit": hit, "traj": traj,
            "dec_up": dec_up, "wm_up": wm_up,
            "t1": t1, "t2": t2, "dist": round(dist, 3),
            "steps": steps, "gave_up": gave_up}
