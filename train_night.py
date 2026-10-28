#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_night.py — 晚上（做梦）：LoRA 合并 + 回放 + 失败状态逼近 V* + LoRA 清空

流程：
  1. 合并 LoRA 进 base（成功经验——睡眠巩固）
  2. 回放预训练数据（决策：规则想象；世界：规则监督——防遗忘）
  3. LoRA 清空（新的一天）
  4. 失败状态（failures.json）：从最后状态 → 想象 → 逼近 V*（决策+世界更新——防过拟合：少量）

用法：
  python train_night.py --device cuda --replay 500 --fail-steps 200
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
    p.add_argument("--replay", type=int, default=500)     # 复习预训练数据
    p.add_argument("--fail-steps", type=int, default=200) # 失败状态想象（防过拟合）
    p.add_argument("--lr", type=float, default=3e-4)
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

    # 2. 回放预训练数据（防遗忘——base 小 lr）
    for prm in list(dec.base.parameters()) + list(world.base.parameters()):
        prm.requires_grad = True
    opt = torch.optim.Adam(list(dec.parameters()) + list(world.parameters()), lr=args.lr)
    for step in range(args.replay):
        t1 = (step * 7.3) % 360 - 180
        t2 = 30 + (step * 3.1) % 120
        # 规则目标（可达）
        a1 = (step * 11.7) % 360 - 180
        a2 = 20 + (step * 5.3) % 130
        tx, ty = env.forward_kinematics(a1, a2)
        obs = obs_tensor(t1, t2, (tx, ty), device)
        d1, d2 = dec(obs)
        # 决策：规则世界想象
        ra1 = (t1 + d1) * math.pi / 180.0
        ra2 = (t1 + d1 + (t2 + d2)) * math.pi / 180.0
        hx = torch.cos(ra1) + torch.cos(ra2)
        hy = torch.sin(ra1) + torch.sin(ra2)
        dist = torch.sqrt((hx - tx) ** 2 + (hy - ty) ** 2 + 1e-6)
        loss_d = dist.mean()
        # 世界：规则监督
        dr = torch.tensor([[rng.uniform(-1, 1), rng.uniform(-1, 1)]], device=device)
        pred = world(obs, dr)
        nt1 = t1 + float(dr[0, 0]) * 60
        nt2 = min(env.ELBOW_MAX, max(env.ELBOW_MIN, t2 + float(dr[0, 1]) * 60))
        nhx, nhy = env.forward_kinematics(nt1, nt2)
        ohx, ohy = env.forward_kinematics(t1, t2)
        raw = torch.tensor([nhx - ohx, nhy - ohy, 0.0, env.pain(nt1, nt2),
                            env.load(nt1, nt2, None)], device=device)
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
            print(f"[replay] {step+1} | 决策距离 {loss_d.item():.3f} | 世界 mse {loss_w.item():.5f}", flush=True)

    # 3. LoRA 清空（新的一天）
    for lora in list(dec.loras().values()) + list(world.loras_dict().values()):
        lora.reset()
    for prm in list(dec.base.parameters()) + list(world.base.parameters()):
        prm.requires_grad = False
    print("[night] LoRA 清空", flush=True)

    # 4. 失败状态：从最后状态 → 想象 → 逼近 V*
    try:
        with open("failures.json", encoding="utf-8") as f:
            fails = json.load(f)
    except FileNotFoundError:
        fails = []
    for prm in list(dec.loras().values()):
        for pp in prm.parameters():
            pp.requires_grad = True
    opt_d = torch.optim.Adam([p for p in dec.parameters() if p.requires_grad], lr=1e-4)
    for fi, fail in enumerate(fails[:20]):
        target = tuple(fail["target"])
        t1, t2 = fail["t1"], fail["t2"]
        vstar = fail["vstar"]
        for step in range(args.fail_steps // max(1, len(fails[:20])) or 1):
            obs = obs_tensor(t1, t2, target, device)
            d1, d2 = dec(obs)
            a1 = (t1 + d1) * math.pi / 180.0
            a2 = (t1 + d1 + (t2 + d2)) * math.pi / 180.0
            hx = torch.cos(a1) + torch.cos(a2)
            hy = torch.sin(a1) + torch.sin(a2)
            dist = torch.sqrt((hx - target[0]) ** 2 + (hy - target[1]) ** 2 + 1e-6)
            nt2 = torch.clamp(t2 + d2, env.ELBOW_MIN, env.ELBOW_MAX)
            pain = torch.relu(nt2 - env.FOLD_PAIN) * 0.5
            v = -dist - 5.0 * (torch.exp(pain / 10.0) - 1.0)
            loss = -v
            opt_d.zero_grad()
            loss.backward()
            opt_d.step()
        print(f"[fail {fi}] 目标 {target} V* {vstar:.3f} | 当前价值 {float(v):.3f}", flush=True)

    torch.save({"dec_base": dec.base.state_dict(),
                "dec_heads": {k: getattr(dec, k).state_dict()
                              for k in ("head_shoulder", "head_elbow")},
                "world_base": world.base.state_dict(),
                "world_heads": world.heads.state_dict()}, "night_base.pt")
    print("[save] night_base.pt（合并+回放+失败修正后的 base）", flush=True)


if __name__ == "__main__":
    main()
