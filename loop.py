#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""loop.py — 决策 / 世界共用的任务循环（train_day.py 调的就是这一个）

为什么要有这个文件：
  这段循环原先在 train_day.py 和单任务脚本里各抄了一份，而且抄歪了——
  单任务脚本有内层重试（一个状态上最多试 300 次），train_day 没有（拒绝一次就烧掉一格 step）。
  于是两边的 step 不是同一个时钟，对比轨迹时得出了一个假 bug（"权重一模一样结果却不同"）。
  现在只有这一份。

步的定义（两边一致）：
  step  = 一次"被接受的移动"
  retry = 同一个状态上最多试几次（不安全的丢掉，安全的记下，过线的立刻执行）

每一步一定会动（除非所有候选都不安全）：试满 max_retries 次仍没有"价值过线"的动作时，
执行安全候选里价值最高的那一个。这是为了对付"目标附近预测误差 > 真实改善"的情况——
以前会原地不动、磨到上限然后失败（实测失败任务里 12/14 是"一步就能到"）。

价值：
  V_new = -dist' - 5·(exp(pain'/10) - 1) - 2·load'
  验证：V_new > V_cur + ACCEPT_EPS 才执行（真实价值 V_cur 用环境真值，不用世界模型）
"""

import math

import torch

import dream
import env

HIT_DIST = 0.1      # 到达判定（米）
ACCEPT_EPS = 0.01   # 价值至少提升这么多才执行（平手不走）
SAFE_PAIN = 1.0     # 安全门：预测痛允许比当前多这么多
SAFE_LOAD = 0.5     # 安全门：预测负载允许比当前多这么多
# 世界更新门控（归一化误差，换算成价值都 ≥ 接受门槛的 4 倍才值得学）
GATE_POS = 0.05     # ×2  = 0.10 米
GATE_PAIN = 0.02    # ×30 = 0.60 分痛
GATE_LOAD = 0.02    # ×1  = 0.02 负载 → 价值 0.04


def obs_tensor(t1, t2, target, device):
    return torch.tensor(env.obs_vector(t1, t2, target), dtype=torch.float32,
                        device=device).unsqueeze(0)


def true_value(t1, t2, target):
    """真实价值（环境真值，不是世界模型预测）。

    痛是状态函数：θ2 越过 160 的阻力点之后才出现，而且越挤越深就越痛
    （所以"已经很痛了还继续推"会真的更痛）。
    """
    hx, hy = env.forward_kinematics(t1, t2)
    dist = math.hypot(hx - target[0], hy - target[1])
    p = env.pain(t1, t2)
    return -dist - 5.0 * (math.exp(min(p, env.PAIN_CAP) / 10.0) - 1.0) - 2.0 * env.load(t1, t2, None)


def world_value(pred_n, hx, hy, target, scale):
    """世界模型预测的价值（run_task 与诊断脚本共用，保证口径一致）。

    pred_n：归一化预测；scale：OUT_SCALE。

    痛/负载都夹到 ≥0：物理上不存在负痛/负负载，而"预测负痛"会在价值里
    凭空造出奖励——痛项是指数的，−0.2 的痛约等于 +0.1 的价值，是接受门槛
    (0.01) 的十倍。实测后果：智能体在目标附近输出 (0,0) 不动，却因为
    "预测自己能少痛一点"每帧都过门，一路刷分到停滞上限。
    缓解仍然能表达：预测痛 < **当前真实痛** 即可（v_cur 用的是真值）。
    """
    pn = pred_n * scale
    nhx, nhy = hx + pn[0, 0], hy + pn[0, 1]
    dist = torch.sqrt((nhx - target[0]) ** 2 + (nhy - target[1]) ** 2 + 1e-6)
    pain = torch.clamp(pn[0, 3], min=0.0, max=env.PAIN_CAP)
    load = torch.clamp(pn[0, 4], min=0.0)
    return -dist - 5.0 * (torch.exp(pain / 10.0) - 1.0) - 2.0 * load


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


def _world_update(world, opt_w, obs, act, pred, o1, o2, t1, t2, pain_a, load_a, scale):
    """执行后用真实结果监督世界模型（门控：只学预测得离谱的地方）。返回是否更新了。

    位置和感觉都用 env.step_body 给的真实结果：θ2 受阻力限制，痛/负载由结果角度决定。
    """
    nhx, nhy = env.forward_kinematics(t1, t2)
    ohx, ohy = env.forward_kinematics(o1, o2)
    raw = torch.tensor([nhx - ohx, nhy - ohy, 0.0, pain_a, load_a], device=scale.device)
    actual = raw / scale
    err = (pred.squeeze(0) - actual).abs()
    # 三项都要查：以前只查位置和痛，负载偏 0.046（价值 0.09，门槛的 9 倍）
    # 却永远触发不了更新——它就是"智能体总觉得自己更累"的来源。
    if (float(err[:2].max()) <= GATE_POS           # 位置
            and float(err[3]) <= GATE_PAIN         # 痛
            and float(err[4]) <= GATE_LOAD):       # 负载
        return False
    pred2 = world(obs, act.detach())
    loss = ((pred2.squeeze(0) - actual) ** 2).mean()
    # 痛项：价值空间相对加权（与预训练/夜间共用 env.pain_loss_t）
    loss = loss + env.pain_loss_t(pred2[0, 3] * scale[3], actual[3] * scale[3])
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
             max_steps=100, max_retries=50, stall_limit=15,
             eval_only=False, verbose=False, oracle_teacher=False):
    """跑一个任务（从 (t1,t2) 出发）。

    每一步的重试规则：
      1. 决策给动作 → 世界模型预测价值；
      2. **安全门**：预测痛/负载显著超过当前 → 不安全，丢掉；
      3. 安全的动作里，有超过 v_cur + 0.01 的 → 立刻执行（早退出，少更新决策）；
      4. 试满 max_retries 次都没有过线的 → **执行安全候选里价值最高的那个**（保证前进）；
      5. 一个安全的都没有 → 才真的放弃。

    为什么要有第 4 条：以前"不过线就原地不动"，于是目标附近只要世界模型的预测
    偏差压过真实改善（噪声 0.1 > 信号 0.03），就只能反复重试到上限然后失败——
    实测 14 个失败里 12 个"一步就能到"。它失败不是因为不会走，是不肯迈第一步。

    eval_only=True：权重全冻结（不更新决策/世界），但第 4 条同样生效。

    oracle_teacher=True：**只把"学习信号"换成真实物理**（规则世界、可微），
    接受/拒绝的判断仍然用世界模型。用来回答一个二选一的问题：
      "决策学坏了，是因为老师（世界模型）太差，还是因为学习规则本身有问题？"
    真实物理的梯度走 dream.decision_loss()——和预训练/夜间复习同一套。

    返回：
      hit      是否到达（dist < HIT_DIST）
      traj     经过的每一帧（含起点和终点）
      dec_up   决策更新次数
      wm_up    世界更新次数
      t1,t2    最终关节角（度）
      dist     最终距离
      steps    被接受的移动次数
      gave_up  是否因为"一个安全的动作都没有"而放弃
    """
    device = next(dec.parameters()).device
    scale = torch.tensor(env.OUT_SCALE, device=device)
    traj = [state_row(0, t1, t2, target)]
    dec_up = wm_up = 0
    steps = 0
    hit = gave_up = False
    dist = traj[0]["dist"]
    v_cur = 0.0
    best_dist = dist       # 已见的最好距离（停滞检测用）
    since_improve = 0

    for step in range(max_steps):
        hx, hy = env.forward_kinematics(t1, t2)
        dist = math.hypot(hx - target[0], hy - target[1])
        if dist < HIT_DIST:
            hit = True
            break
        v_cur = true_value(t1, t2, target)
        cur_pain = env.pain(t1, t2)
        cur_load = env.load(t1, t2, None)
        obs = obs_tensor(t1, t2, target, device)   # 重试期间状态不变 → 只算一次

        best = None      # 安全候选里价值最高的：(v, d1, d2)
        chosen = None    # 要执行的动作：(d1, d2, v, retry)
        tries = 0
        for retry in range(max_retries):
            tries = retry + 1
            d1, d2 = dec(obs)
            act = torch.cat([d1, d2], dim=-1) / 60.0
            pred = world(obs, act)
            v_new = world_value(pred, hx, hy, target, scale)
            pn = pred * scale
            # 安全门：不能显著更痛/更吃力（距离变差不拦——下一步能纠）
            safe = (float(pn[0, 3]) <= cur_pain + SAFE_PAIN
                    and float(pn[0, 4]) <= cur_load + SAFE_LOAD)
            if safe:
                if best is None or float(v_new) > best[0]:
                    best = (float(v_new), float(d1), float(d2))
                if float(v_new) > v_cur + ACCEPT_EPS:      # 过线：立刻执行
                    chosen = (float(d1), float(d2), float(v_new), retry)
                    break
            if eval_only:          # 冻结时不更新 → 原地再试也没意义
                break
            if oracle_teacher:
                # 老师换成真实物理：梯度按"规则世界里的真实价值"给
                loss_d, _v = dream.decision_loss(dec, obs, t1, t2, target)
            else:
                loss_d = -v_new    # 拒绝 → 决策往上抬（用的是世界模型的判断）
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()
            dec_up += 1

        if chosen is None:
            if best is None:       # 试了这么多次，一个安全的动作都没有 → 真的放弃
                gave_up = True
                if verbose:
                    print(f"[step {step}] {tries} 次尝试全不安全 → 放弃 | "
                          f"v_cur {v_cur:.3f} | dist {dist:.2f}", flush=True)
                break
            # 没有动作过线，但至少有一个安全的 → 执行其中价值最高的，保证前进
            chosen = (best[1], best[2], best[0], tries - 1)
            if verbose:
                print(f"[step {step}] 无动作过线 | 执行最优安全动作 v {best[0]:.3f} | "
                      f"v_cur {v_cur:.3f} | dist {dist:.2f} | 重试 {chosen[3]}", flush=True)
        elif verbose:
            print(f"[step {step}] 过线 | v_new {chosen[2]:.3f} | v_cur {v_cur:.3f} | "
                  f"dist {dist:.2f} | 重试 {chosen[3]}", flush=True)

        # 执行选中的动作
        d1, d2 = chosen[0], chosen[1]
        o1, o2 = t1, t2
        t1, t2, _want2, pain_a, load_a = env.step_body(t1, t2, d1, d2)
        if not eval_only:
            act = torch.tensor([[d1 / 60.0, d2 / 60.0]], device=device)
            with torch.no_grad():
                pred = world(obs, act)
            if _world_update(world, opt_w, obs, act, pred, o1, o2, t1, t2,
                             pain_a, load_a, scale):
                wm_up += 1

        steps = step + 1
        traj.append(state_row(steps, t1, t2, target))

        nhx, nhy = env.forward_kinematics(t1, t2)
        ndist = math.hypot(nhx - target[0], nhy - target[1])
        if ndist < best_dist - 1e-6:      # 真的更近了
            best_dist = ndist
            since_improve = 0
        else:
            since_improve += 1
            if since_improve >= stall_limit:   # 原地打转 → 停
                gave_up = True
                if verbose:
                    print(f"[step {step}] 连续 {since_improve} 步没有更接近"
                          f"（{ndist:.2f} 米）→ 停止", flush=True)
                break

    # 最终距离按现在的 (t1,t2) 重算（循环跑满时 dist 还停在最后一次移动之前）
    hx, hy = env.forward_kinematics(t1, t2)
    dist = math.hypot(hx - target[0], hy - target[1])
    return {"target": tuple(target), "hit": hit, "traj": traj,
            "dec_up": dec_up, "wm_up": wm_up,
            "t1": t1, "t2": t2, "dist": round(dist, 3),
            "steps": steps, "gave_up": gave_up}
