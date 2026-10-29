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
### 架构已定（依据见 [DESIGN 第十节](DESIGN.md)）

| | 做法 | 一句话依据 |
|---|---|---|
| **白天** | base 冻结；决策 + 世界 LoRA **都累积**；world-lr 1e-4 | 只累积决策、世界冻住 → 留出集 79.4%；都累积 → 92.8% |
| **晚上** | 合并 → 回放（lr 3e-5）→ LoRA 归零 | 3e-4 回放会把合并成果擦掉；3e-5 才留得住 |

已经排除、别再试：成功才提交（更差）、每任务清零（经验留不下来）、`--world-lr 1e-2`（悲观崩）。

**判断标准不是"准"，是"世界变了能不能调整"。** 目前准确率最高的配置是 oracle 老师（95.6%），
而它就是"不要世界模型、直接训练决策"——所以世界模型的价值在**提供"和预期不符"的信号**，不在预测得准。

## 流程

```bash
# ⓪ 先钉住任务集（一次性；之后所有对比都跑同一个文件）
python tool/make_tasks.py --seed 1 --draws 100 --out tasks/seed1.json
python tool/make_tasks.py --seed 2 --draws 100 --out tasks/seed2.json

# ① 做梦（一次性，10k 步）
python pretrain_lora.py --device cuda --steps 10000

# ② 生活（白天）——对比实验必须用 --tasks-file，--out 分开存
#    --ckpt-out 让每次实验的适配器单独落盘（否则都会覆盖 lora_day.pt）
python train_day.py --device cuda --tasks-file tasks/seed1.json \
    --out runs/s1.json --ckpt-out runs/s1_acm.pt
python train_day.py --device cuda --tasks-file tasks/seed1.json \
    --commit-on-success --out runs/s1_commit.json               # 只保留成功任务的改动
python train_day.py --device cuda --tasks-file tasks/seed1.json \
    --commit-on-success --commit-dec-only \
    --out runs/s1_commitdec.json                                 # 同上，但世界的修正保留
python train_day.py --device cuda --tasks-file tasks/seed1.json \
    --per-task-reset --world-lr 1e-2 --out runs/s1_adapt.json   # 每任务清零（旧版行为）
python train_day.py --device cuda --target -1.0 -1.0             # 单任务（自动逐步打印）
python train_day.py --device cuda --tasks-file tasks/seed1.json --eval-only \
    --base pretrain_base.pt --out runs/s1_frozen.json            # 冻结评估

# 保持性测试：拿一天结束后的适配器**冻结**再跑一遍，按顺序分块看早期任务掉不掉
python train_day.py --device cuda --tasks-file tasks/seed1.json --eval-only \
    --base runs/s1_acm.pt --load-dec-lora --no-save --out runs/s1_final.json
python tool/retention.py runs/s1_frozen.json runs/s1_final.json --blocks 4

# ③ 睡眠
python train_night.py --device cuda --replay 2000 --lr 3e-5      # ⚠️ 必须是微调级 lr

# ④ 睡后对比（同一批任务，配对检验）
python tool/compare_eval.py runs/s1.json runs/s1_night.json
python tool/compare_steps.py runs/s1.json runs/s1_night.json     # 步数（比完成率灵敏）
```

**为什么一定要 `--tasks-file`**：`--tasks N` 是"抽 N 次"（不可达的丢掉），同一个命令两次能抽到不同的任务集；再加上 LoRA 初始化用的是全局 RNG，两次能差 ±2 个任务。固定任务集 + `--seed` 之后，同一命令剩下的差异只有 CUDA 非确定性。

`tasks/seed1.json` 是 `--seed 1 --draws 100` 抽出来的 87 个任务，数量和 README 里那批一致（但历史那次没存任务列表，无法保证逐点相同）。三份种子集都是 i.i.d. 抽取（同分布）：seed1 87 个、seed2 90 个、seed3 90 个；彼此最接近的点对约 2 cm（1~5 对在 5 cm 内），**不是严格不重叠**——所以统计时要靠配对，不要靠"任务一定没见过"。

世界可以变（测"能不能调整"，见 [DESIGN 第十一节](DESIGN.md)）：

```bash
python train_day.py --device cuda --tasks-file tasks/seed1.json \
    --body d2_bias=3 --out runs/s1_body.json --ckpt-out runs/s1_body.pt
python tool/check_world.py --device cuda --body d2_bias=3 \
    --ckpts pretrain_base.pt runs/s1_body.pt
```

`--body` 可给多个：`d1_bias` / `d2_bias` / `fold_start` / `fold_yield` / `pain_start`。
**预训练永远用默认值**——先在世界 A 里学会，才谈"世界变成 B 以后能不能调回来"。

## 工具（`tool/`）

| 工具 | 用途 |
|---|---|
| `make_tasks.py` | **生成固定的任务集**（`tasks/seed1.json`）——对比实验的基准 |
| `check_stall.py` | 失败状态分解：预测 vs 真实的距离/痛/负载 |
| `check_policy.py` | 误拒/误接率（按动作大小分桶） |
| `check_body.py` | 标量身体 `step_body` 与可微版 `step_theta2_t` 是否一致 |
| `check_lora.py` | 一份 LoRA 把模型改了多少（权重范数 + 动作/预测变化） |
| `check_world.py` | 世界模型准不准：位置/痛/价值误差 + 价值判定翻转率（冻结评估量不到它） |
| `compare_eval.py` | 两组完成度的配对检验（McNemar） |
| `compare_steps.py` | 两组步数的配对检验（符号检验） |
| `retention.py` | 按任务顺序分块，看早期任务有没有被忘掉（需要两份冻结评估） |
| `show_fails.py` | 失败任务列表：类型 / 到肩距离 / 需要 θ2 / 会痛多少（`--summary` 只看统计） |
| `view.html` | 把 `trajectory.json` 拖进浏览器看动画 |

## 结果（实测，固定任务集 `tasks/seed1.json`，87 个任务）

| 配置 | 完成度 | 步数 |
|---|---|---|
| 冻结 `pretrain_base`（`--eval-only`） | 88.5%（77/87） | 3.75 |
| 每任务适应（`--per-task-reset --world-lr 1e-2`） | **94.3%**（82/87） | 3.67 |
| 默认累积（不重置） | 92.0%（80/87） | — |
| 成功才提交（`--commit-on-success`） | 83.9%（73/87） | — |

配对检验：
- 冻结 → 每任务适应：完成率 McNemar **p=0.18（不显著）**；步数符号检验 **p=0.039（8 省 1 多）**。
  **真正稳的信号是步数，不是完成率**——两者都成功后，"走多少步"才区分得出适应有没有用。
- 累积 → 成功才提交：B **少 7 个任务**（换输 9 / 换赢 2，p=0.065）。

**"失败任务的改动会带偏模型"这个假设，至少对决策那头不成立。** 整套回滚反而更差，
但这里混着两个原因：决策的修正被丢掉了，**世界的真实偏差修正也被丢掉了**。
`--commit-dec-only`（只回滚决策、保留世界）就是为了把这两个原因拆开——**还没测**。

上表的 67.8% / 54.0% 是旧代码的数字（在"负痛夹到 ≥0、θ1 环绕、痛损失改价值权重、
门控收紧"之前测的），**不能再引用**。

> 另一个数字值得记住：一天累积的 LoRA 改了 `||ΔW||/||W||` 的 44~61%、动作平均变 6°，
> 但**冻结之后只值 +1 个任务**（88.5% → 89.7%，p=1.00）。白天得到的是"不一样的策略"，
> 不是"更好的策略"。

> 旧数据：累积决策 LoRA 67.8%、累积世界 LoRA（`--keep-world`）54.0%——固定任务集之前、旧代码。
> 睡眠（合并 + 2000 步回放 @ lr 3e-5）在另一批 46 个任务上：60.9% → 80.4%（换赢 9 / 换输 0，p=0.004）。

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
7. 单次运行仍有 **CUDA 非确定性**（`--seed` 现在会固定任务集和权重初始化，但同一个种子
   同一个种子两次跑的更新次数仍可能差几点）。

## 参考

- Dreamer（model-based RL）
- IQL/D4RL（离线 RL）
- Proprioceptive Learning（arXiv:2308.08538，本体感觉）
- Pain Axis（arXiv:2609.16247，LLM 疼痛方向）
- Silicopathy（IEEE，人工疼痛/共情）

## 怎么量"白天到底学到了什么"

一个容易犯的错：白天在 seed1 上训练，然后拿 seed1 考它。那是**拟合**的尺子。
真正的学习要考**没见过的目标**：

```
白天：只在 seed1 上下功（87 个）
考试：冻结后考 seed2 + seed3（180 个从没训练过的任务）
```

```bash
# 基准：不训练，直接考留出集
python train_day.py --device cuda --tasks-file tasks/seed2.json tasks/seed3.json \
    --eval-only --base pretrain_base.pt --out runs/heldout_base.json

# 白天在 seed1 学，冻结后考留出集（--ckpt-out 保住适配器）
python train_day.py --device cuda --tasks-file tasks/seed1.json \
    --out runs/s1_acm.json --ckpt-out runs/s1_acm.pt
python train_day.py --device cuda --tasks-file tasks/seed2.json tasks/seed3.json \
    --eval-only --base runs/s1_acm.pt --load-dec-lora --no-save --out runs/heldout_acm.json
python tool/compare_eval.py runs/heldout_base.json runs/heldout_acm.json
```

2×2（决策 × 世界，哪一个累积才让模型真的学到东西）：

| 命令 | 决策 | 世界 |
|---|---|---|
| `--per-task-reset` | 重置 | 重置 |
| `--per-task-reset dec` | 重置 | 累积 |
| `--per-task-reset world` | 累积 | 重置 |
| （默认） | 累积 | 累积 |

存档时会把**不参与累积的那一侧清零**——`--per-task-reset dec` 跑完时决策 LoRA 里
只剩最后一个任务的临时适配，留着它，冻结评估就会把这份临时适配当成"白天学到的决策"。

### 实测（白天在 seed1 学，考试在留出的 seed2+seed3）

| 条件 | 留出集 | 相对基准 |
|---|---|---|
| 不训练（基准） | 92.2%（166/180） | — |
| 决策 + 世界都累积 | 92.8%（167/180） | +1（p=1.00） |
| **只有世界累积**（决策每任务重置） | 92.2%（166/180） | **逐位相同** |
| **只有决策累积**（世界每任务重置） | 79.4%（143/180） | **−23（p=0.0003）** |
| 同上，但学习信号换成**真实物理**（`--oracle-teacher`） | **95.6%（172/180）** | **+6（p=0.18）** |

两个要点：

1. **"只有世界累积"这一格是死实验。** 冻结评估（`--eval-only`）每个状态只问决策一次，
   而"接受"和"没接受"执行的是同一个动作——世界模型唯一的影响是安全门。
   它没有翻转任何一次安全判定，于是 180 个任务逐位相同（0 胜 0 负）。
   想量世界模型，用 `tool/check_world.py`，别用完成度。
2. 真正的结论是**老师质量**：决策累积的伤害只在世界冻结时出现，而换成真实物理
   （`--oracle-teacher`，行为路径不变、只换学习信号）后 **79.4% → 95.6%（+29，p<1e−4）**。
   同一个流程、同样约 9.5k 次更新、同样"只从拒绝里学"——唯一变的是梯度来自真值还是世界模型。

   所以那条链是：**老师准 → 白天帮忙；老师一般但一直在动 → 白天中性；老师准但被冻住 → 白天帮倒忙**。
   "自我校准"这个故事成立，但**当前一天只能让世界准确率动 2%**（位置 0.2145→0.2076 米），
   差得远，不足以改变符号。

   天花板也不高：即使老师完美，87 个任务学一天也只值 **+6（p=0.18）**。
   而且真实物理的梯度走的正是 `dream.decision_loss`——和预训练同一套损失，
   而白天任务全从 (30°,90°) 出发，预训练却没在这个角落采够。所以这 +6 可能是**补课**，不是"经验"。

### 世界学习率的稳定边界

同一套更新规则，步长小是改善、大步是灾难：

| `--world-lr` | 负载误差 | 价值误差 | 判定翻转 | 留出集 |
|---|---|---|---|---|
| 1e-4（默认） | 0.028 → 0.023 | 0.333 → 0.327 | 8.9% → 8.4% | 92.8% |
| **1e-2（100×）** | 0.028 → **0.111** | 0.333 → **0.519** | 8.9% → **13.9%** | **44.4%** |

1e-2 的崩法有签名：世界从"大致准"偏成**悲观**（误接 115 → 25），白天 38 个失败里
27 个是"全不安全（一步没动）"——安全门被过高的负载预测堵死；
冻结之后 100 个失败里 99 个是**停滞**——决策已经变成"到处乱走、不肯靠近"的策略。
两段伤害不同，别混为一谈。
