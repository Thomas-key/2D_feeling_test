# 2D 感觉实验 — 做梦（预训练）+ 真实微调

## 这是什么

一个 **2D 两关节手臂** 的具身控制实验平台：让"决策模型"通过**感觉**（痛、触觉、负载）学控制手臂——并模仿人类的**睡眠/生活**节奏：

```
睡觉（预训练）：规则世界当老师——做梦（想象）学通用能力
生活（真实）：真实环境微调——小 lr——达标才保存 best（不忘记）
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

```
env.py               2D 手臂环境（正/逆运动学、痛/触觉/负载、感觉网格 48 维）
train_world_num.py   世界模型预训练（规则数据监督：动作 → 状态）
pretrain_decision.py 决策预训练（规则世界——想象梯度）
train_exec.py        真实微调（多次——验证-执行、小 lr、达标保存 best）
closed_loop_exec.py  真实微调（单次——带在线更新）
render_html.py       轨迹可视化（HTML 动画）
check_world_num.py   世界模型验证（敏感度 + 准确度）
```

## 流程

```bash
# 睡觉（预训练）
python train_world_num.py --device cuda --steps 10000     # 世界模型
python pretrain_decision.py --device cuda --steps 10000   # 决策

# 生活（真实微调）
python train_exec.py --device cuda --episodes 2000        # 多次
python closed_loop_exec.py --target -1.0 -1.0 --device cuda  # 单次
```

## 结果（示例）

- 目标 (−1.0,−1.0)：23 步到达（距离 0.07）
- 目标 (−0.5,−1.0)：11 步到达（距离 0.07）

## 局限（诚实）

1. **玩具环境**（2D 手臂——已知规则——不是真实机器人）；
2. **技术组件已知**（见上表）——贡献在组合/理解；
3. 决策/世界模型规模小（MLP）——不是基础模型；
4. 感觉是**功能层面**（不是主观体验）。

## 参考

- Dreamer（model-based RL）
- IQL/D4RL（离线 RL）
- Proprioceptive Learning（arXiv:2308.08538，本体感觉）
- Pain Axis（arXiv:2609.16247，LLM 疼痛方向）
- Silicopathy（IEEE，人工疼痛/共情）
