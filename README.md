# 2D 感觉实验 — 做梦（预训练）+ 真实微调

## 这是什么

一个 **2D 两关节手臂** 的具身控制实验平台：让"决策模型"通过**感觉**（痛、触觉、负载）学控制手臂——并模仿人类的**睡眠/生活**节奏：

```
做梦（预训练）：规则世界当老师——想象里学通用能力（base 全量训练）
生活（白天）  ：真实验证-执行——拒绝才调整（LoRA，小改）
睡眠（巩固）  ：LoRA 合并进 base → 低 lr 回放 →（可选）失败状态修正
```

## 与现有工作的关系（诚实定位）

**本项目的技术组件大多不是新的**——我们做的是**组合与理解**（教学实验平台），不是 SOTA 贡献：

| 我们的做法 | 对应现有工作 | 差异 |
|---|---|---|
| 做梦（决策想象：世界模型→价值→梯度） | Model-based RL（Dreamer/PlaNet） | 我们**规则世界当老师**（已知动力学——简化/教学） |
| 真实微调（小 lr/达标保存 best） | 经验回放/离线 RL（IQL/D4RL） | 我们用"现实=小调、梦=大调"的睡眠类比 |
| 感觉网格（触觉/负载/痛） | 触觉/本体感觉（Proprioceptive Learning） | 我们组合"痛+触觉+负载"驱动控制 |
| 痛驱动决策（痛惩罚/保护） | 人工痛/情绪（Pain Axis/Silicopathy） | 我们做**具身控制**（不是语言生成） |
| 验证-执行（价值验证→执行/拒绝） | Constrained RL / 安全层 | 我们用"感觉"（痛）做保护 |

**定位**：理解"感觉如何影响具身控制"的**实验平台**——不是 SOTA，不声称技术全新。

## 结构

根目录只放**整体流程**（4 个核心模块 + 3 个阶段脚本）：

| 核心模块 | 作用 |
|---|---|
| `env.py` | 2D 手臂环境：运动学、痛/触觉/负载、折叠阻力（160° 之后是阻力不是墙）、感觉网格 48 维 |
| `lora.py` | LoRA 适配器（A/B）+ 决策 2 头 / 世界 5 头 |
| `loop.py` | 任务循环：安全门 → 重试 → 过线执行 / 试满取最优 → 执行后更新世界。**所有脚本共用这一份** |
| `dream.py` | 规则世界的训练数据与损失（预训练与夜间复习共用同一套分布） |

| 阶段 | 脚本 | 产物 |
|---|---|---|
| ① 做梦 | `pretrain_lora.py` | `pretrain_base.pt` |
| ② 生活 | `train_day.py` | `lora_day.pt`（决策/世界 LoRA + 白天记录） |
| ③ 睡眠 | `train_night.py` | `night_base.pt` |

诊断与单任务工具都在 `tool/`（见下表）。

## 流程

```bash
# ① 做梦（一次性，10k 步）
python pretrain_lora.py --device cuda --steps 10000

# ② 生活（白天）
python train_day.py --device cuda --tasks 100                    # 多任务 → 完成度
python train_day.py --device cuda --target -1.0 -1.0             # 单任务（自动逐步打印）
python train_day.py --device cuda --tasks 100 --eval-only --base pretrain_base.pt  # 冻结评估

# ③ 睡眠
python train_night.py --device cuda --replay 2000 --lr 3e-5      # ⚠️ 必须是微调级 lr

# ④ 睡后对比（同一批任务，配对检验）
python tool/compare_eval.py day_before.json day_after.json
python tool/compare_steps.py day_before.json day_after.json      # 步数（比完成率灵敏）
```

## 工具（`tool/`）

| 工具 | 用途 |
|---|---|
| `closed_loop_exec.py` | 单任务逐步打印（调同一个 `run_task`） |
| `check_models.py` | 决策方向 / 世界位置误差 / 折叠痛预测 |
| `check_stall.py` | 失败状态分解：预测 vs 真实的距离/痛/负载 |
| `check_policy.py` | 误拒/误接率（按动作大小分桶） |
| `check_body.py` | 标量身体 `step_body` 与可微版 `step_theta2_t` 是否一致 |
| `check_lora.py` | 一份 LoRA 把模型改了多少（权重范数 + 动作/预测变化） |
| `compare_eval.py` | 两组完成度的配对检验（McNemar） |
| `compare_steps.py` | 两组步数的配对检验（符号检验） |
| `split_eval.py` | 拆成"训练过的 / 没见过的"任务 |
| `show_fails.py` | 失败任务列表：类型 / 到肩距离 / 需要 θ2 / 会痛多少 |
| `check_reject.py` / `check_enum.py` / `check_small.py` | 单状态的误拒、动作枚举、小动作精度 |
| `view.html` | 把 `trajectory.json` 拖进浏览器看动画 |

## 结果（实测，同一批 87 个任务）

| 配置 | 完成度 |
|---|---|
| 冻结 `pretrain_base` | 88.5% |
| 白天每任务适应（决策重置 + `--world-lr 1e-2`） | **95.4%** |
| 累积决策 LoRA（不重置） | 67.8% |
| 累积世界 LoRA（`--keep-world`） | 54.0%（世界漂成"疑病症"→ 全被安全门拦） |

睡眠（合并 + 2000 步回放 @ lr 3e-5）在另一批 46 个任务上：**60.9% → 80.4%**（换赢 9 / 换输 0，p=0.004）。

**关键教训**：回放的 lr 必须用微调级（3e-5）。用预训练的 3e-4，仅仅 200 步就能把合并的成果从 73.9% 擦到 56.5%。

## 局限（诚实）

1. **玩具环境**（2D 手臂——已知规则——不是真实机器人）；
2. **技术组件已知**（见上表）——贡献在组合/理解；
3. 决策/世界模型规模小（MLP）——不是基础模型；
4. 感觉是**功能层面**（不是主观体验）；
5. **痛区目标（到肩 < 0.35 米）在安全门下不可达**——需要折过 160°，那是设计上"不该去"的地方，
   不是缺陷；但 `vstar()` 仍把它标成"可达"，所以完成度会被低估；
6. 白天的**世界学习很慢**（LoRA 的 A 初始化只有 0.01，有效步长被压到 2e-6/步），
   不放大 `--world-lr` 时几乎没有效果；
7. 单次运行有 **±2 个任务**的波动——`torch` 的随机种子尚未固定（LoRA 初始化用全局 RNG）。

## 参考

- Dreamer（model-based RL）
- IQL/D4RL（离线 RL）
- Proprioceptive Learning（arXiv:2308.08538，本体感觉）
- Pain Axis（arXiv:2609.16247，LLM 疼痛方向）
- Silicopathy（IEEE，人工疼痛/共情）
