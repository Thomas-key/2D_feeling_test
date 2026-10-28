#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_night.py — 晚上（睡眠巩固）：LoRA 合并进 base + 回放 + 失败状态逼近 V*

流程：
  1. 合并 LoRA 进 base（成功经验——睡眠巩固），LoRA 归零
  2. 回放预训练数据（决策：规则想象；世界：规则监督——防遗忘）
  3. 失败状态（failures.json）：从最后状态 → 想象 → 逼近 V*

全程 LoRA 冻在 0：睡觉 = 大幅改长期权重（base），生活 = 小幅改 LoRA。
所以这里优化器里只有 base/heads，不出现 A/B；存档里 LoRA 一定是空的。

用法：
  python train_night.py --device cuda --replay 2000 --fail-steps 4000
"""

import argparse
import json
import math
import random

import torch
import torch.nn.functional as F

import env
from lora import DecisionLoRA, WorldLoRA


def obs_tensor(t1, t2, target, device):
    return torch.tensor(env.obs_vector(t1, t2, target), dtype=torch.float32,
                        device=device).unsqueeze(0)


def load_day(dec, world, path="lora_day.pt"):
    ckpt = torch.load(path, map_location="cpu")
    dec.base.load_state_dict(ckpt["dec_base"])
    for k in ("head_shoulder", "head_elbow"):
        getattr(dec, k).load_state_dict(ckpt["dec_heads"][k])
    world.base.load_state_dict(ckpt["world_base"])
    world.heads.load_state_dict(ckpt["world_heads"])
    for k, lora in dec.loras().items():
        lora.load_state_dict(ckpt["dec_lora"][k])
    for k, lora in world.loras_dict().items():
        lora.load_state_dict(ckpt["world_lora"][k])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--replay", type=int, default=2000)    # 复习预训练数据
    p.add_argument("--fail-steps", type=int, default=4000)  # 失败状态（200/状态）
    p.add_argument("--lr", type=float, default=3e-4, help="回放（base 的学习率）")
    p.add_argument("--fail-lr", type=float, default=1e-4, help="失败状态修正（base 的学习率）")
    args = p.parse_args()
    device = torch.device(args.device)
    rng = random.Random(0)

    dec = DecisionLoRA().to(device)
    world = WorldLoRA().to(device)
    load_day(dec, world)

    # 1. 合并 LoRA 进 base
    for lora in dec.loras().values():
        lora.merge()
    for lora in world.loras_dict().values():
        lora.merge()
    print("[night] LoRA 合并进 base", flush=True)

    # 2. 回放预训练数据（防遗忘——只动 base：睡觉=改长期权重）
    base_params = (list(dec.base.parameters()) + list(dec.head_shoulder.parameters())
                   + list(dec.head_elbow.parameters())
                   + list(world.base.parameters()) + list(world.heads.parameters()))
    for prm in base_params:
        prm.requires_grad = True
    for lora in list(dec.loras().values()) + list(world.loras_dict().values()):
        for prm in (lora.A, lora.B):
            prm.requires_grad = False          # 睡眠期间不碰 LoRA（冻死在 0）
    opt = torch.optim.Adam(base_params, lr=args.lr)
    for step in range(args.replay):
        t1 = (step * 7.3) % 360 - 180
        t2 = 30 + (step * 3.1) % 120
        # 规则目标（可达）
        a1 = (step * 11.7) % 360 - 180
        a2 = 20 + (step * 5.3) % 130
        tx, ty = env.forward_kinematics(a1, a2)
        obs = obs_tensor(t1, t2, (tx, ty), device)
        d1, d2 = dec(obs)
        # 决策：规则世界想象（价值——距离+痛(指数)——与预训练一致）
        want2 = t2 + d2
        t2n = torch.clamp(want2, env.ELBOW_MIN, env.FOLD_LIMIT)   # 身体：最多到 160
        ra1 = (t1 + d1) * math.pi / 180.0
        ra2 = ra1 + t2n * math.pi / 180.0
        hx = torch.cos(ra1) + torch.cos(ra2)
        hy = torch.sin(ra1) + torch.sin(ra2)
        dist = torch.sqrt((hx - tx) ** 2 + (hy - ty) ** 2 + 1e-6)
        pain = env.pain_t(want2)        # 感觉：按"想去的角度"算（越推越痛）
        v = -dist - 5.0 * (torch.exp(pain / 10.0) - 1.0)
        loss_d = -v.mean()
        # 世界：规则监督
        dr = torch.tensor([[rng.uniform(-1, 1), rng.uniform(-1, 1)]], device=device)
        pred = world(obs, dr)
        # 位置被身体挡住；痛/负载按"想去的角度"算（痛来自发力）
        nt1, nt2, _w2, pain_a, load_a = env.step_body(
            t1, t2, float(dr[0, 0]) * 60, float(dr[0, 1]) * 60)
        nhx, nhy = env.forward_kinematics(nt1, nt2)
        ohx, ohy = env.forward_kinematics(t1, t2)
        raw = torch.tensor([nhx - ohx, nhy - ohy, 0.0, pain_a, load_a], device=device)
        scale = torch.tensor(env.OUT_SCALE, device=device)
        actual = raw / scale
        err = (pred.squeeze(0) - actual) ** 2
        loss_w = err.mean()
        pain_pred = torch.exp(torch.clamp(pred[0, 3] * 30.0, -30.0, 60.0) / 10.0)
        pain_act = torch.exp(torch.clamp(actual[3] * 30.0, -30.0, 60.0) / 10.0)
        loss_w = loss_w + 0.5 * (pain_pred - pain_act) ** 2
        loss = loss_d + loss_w
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 200 == 0:
            print(f"[replay] {step+1} | 决策价值 {float(v):.3f} | 世界 mse {loss_w.item():.5f}", flush=True)

    # 3. LoRA 保持清零（新的一天）——整晚没训练过它，这里再显式确认一次
    for lora in list(dec.loras().values()) + list(world.loras_dict().values()):
        lora.reset()
    for prm in list(dec.base.parameters()) + list(world.base.parameters()):
        prm.requires_grad = False
    print("[night] LoRA 清零（睡眠期间不碰）", flush=True)

    # 4. 失败状态：从最后状态 → 想象 → 逼近 V*
    try:
        with open("failures.json", encoding="utf-8") as f:
            fails = json.load(f)
    except FileNotFoundError:
        fails = []
    # 失败反思也改 base（夜里只动长期权重，LoRA 仍冻在 0）
    dec_base_params = (list(dec.base.parameters()) + list(dec.head_shoulder.parameters())
                       + list(dec.head_elbow.parameters()))
    world_base_params = list(world.base.parameters()) + list(world.heads.parameters())
    for prm in dec_base_params + world_base_params:
        prm.requires_grad = True
    opt_d = torch.optim.Adam(dec_base_params, lr=args.fail_lr)
    opt_w2 = torch.optim.Adam(world_base_params, lr=args.fail_lr)
    for fi, fail in enumerate(fails[:20]):
        target = tuple(fail["target"])
        t1, t2 = fail["t1"], fail["t2"]
        vstar = fail["vstar"]
        # 世界：失败状态 + 批量动作（±2~60）→ 规则 → 监督
        acts = []
        for d in (2, 5, 10, 30, 60):
            for sgn in (-1, 1):
                acts.append((sgn * d, 0))
                acts.append((0, sgn * d))
        for d1, d2 in acts:
            obs = obs_tensor(t1, t2, target, device)
            act = torch.tensor([[d1 / 60.0, d2 / 60.0]], device=device)
            pred = world(obs, act)
            nt1, nt2, _w2, pain_a, load_a = env.step_body(t1, t2, d1, d2)
            nhx, nhy = env.forward_kinematics(nt1, nt2)
            ohx, ohy = env.forward_kinematics(t1, t2)
            raw = torch.tensor([nhx - ohx, nhy - ohy, 0.0, pain_a, load_a], device=device)
            scale = torch.tensor(env.OUT_SCALE, device=device)
            actual = raw / scale
            diff = pred.squeeze(0) - actual
            w_pos = 1.0 / (actual[:2].abs() + 0.05) ** 2
            loss_w = (diff[:2] ** 2 * w_pos).mean() + (diff[2:] ** 2).mean()
            pp = torch.exp(torch.clamp(pred[0, 3] * 30.0, -30.0, 60.0) / 10.0)
            pa = torch.exp(torch.clamp(actual[3] * 30.0, -30.0, 60.0) / 10.0)
            loss_w = loss_w + 0.5 * (pp - pa) ** 2
            opt_w2.zero_grad()
            loss_w.backward()
            opt_w2.step()
        # 决策：失败状态单步最大价值（200/状态）
        for step in range(args.fail_steps // max(1, len(fails[:20])) or 1):
            obs = obs_tensor(t1, t2, target, device)
            d1, d2 = dec(obs)
            want2 = t2 + d2
            t2n = torch.clamp(want2, env.ELBOW_MIN, env.FOLD_LIMIT)   # 身体：最多到 160
            a1 = (t1 + d1) * math.pi / 180.0
            a2 = a1 + t2n * math.pi / 180.0
            hx = torch.cos(a1) + torch.cos(a2)
            hy = torch.sin(a1) + torch.sin(a2)
            dist = torch.sqrt((hx - target[0]) ** 2 + (hy - target[1]) ** 2 + 1e-6)
            pain = env.pain_t(want2)    # 感觉：按"想去的角度"算
            v = -dist - 5.0 * (torch.exp(pain / 10.0) - 1.0)
            loss = -v
            opt_d.zero_grad()
            loss.backward()
            opt_d.step()
        print(f"[fail {fi}] 目标 {target} V* {vstar:.3f} | 当前价值 {float(v):.3f}", flush=True)

    # 存档前再确认一次：睡完 LoRA 必须是空的（否则 base 会带着夜里的小尾巴）
    for lora in list(dec.loras().values()) + list(world.loras_dict().values()):
        lora.reset()
    torch.save({"dec_base": dec.base.state_dict(),
                "dec_heads": {k: getattr(dec, k).state_dict()
                              for k in ("head_shoulder", "head_elbow")},
                "dec_lora": {k: v.state_dict() for k, v in dec.loras().items()},
                "world_base": world.base.state_dict(),
                "world_heads": world.heads.state_dict(),
                "world_lora": {k: v.state_dict() for k, v in world.loras_dict().items()}},
               "night_base.pt")
    print("[save] night_base.pt（合并+回放+失败修正后的 base）", flush=True)


if __name__ == "__main__":
    main()
