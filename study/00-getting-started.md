# 00 · 环境配置与基础概念

> 原文：<https://datawhalechina.github.io/agentic-rl-lab/experiments/getting-started/>
> 本地：`experiments/getting-started/readme.md` · 阅读日期：2026-11-14
> 对应 Datawhale 组队学习 Task01

## 1. 一句话结论

整个 Lab 的算力形态是「**本地写训练循环 + 云端跑梯度**」：PyTRIO 当远程后训练服务，SwanLab 当实验记录，
仓库代码只负责数据、tokenizer、reward 和循环。所以**一台能联网的笔记本就够，全程不需要本地 GPU**。

## 2. 谁在干什么

| 组件 | 跑在哪 | 负责 |
| --- | --- | --- |
| PyTRIO `0.2.9` | 云端 | 前向、反向、优化器更新、采样、权重保存 |
| SwanLab `0.10.4` | 云端 + 本地日志 | loss/reward 曲线、超参、终端输出 |
| 本仓库代码 | 本地 | 数据集、tokenizer、reward、训练循环 |

自己补两句关键的边界条件：

- 本地 torch 是 **CPU 版**（`pyproject.toml` 里 pin 了 `pytorch-cpu` 源）。**不要换 CUDA 版**。
- torch 只在**写自定义损失**时才真正参与计算：
  - `cross_entropy` / `importance_sampling` / `ppo` 这类内置 loss → 云端算，本地 torch 不参与；
  - 走 `forward_backward_custom`（grpo 的 cispo、DAPO、GSPO 等）→ loss 的 autograd 在本地跑，这时才用到 torch。
  - 仓库把 torch 塞进基础依赖，是因为多个章节的脚本文件头就 `import torch`；自己写脚本且只用内置 loss 可以省掉。
- 前置只需三样：Python ≥ 3.13（uv 管理） + PyTRIO 账号 + SwanLab 账号。

## 3. 动手记录（命令清单）

```bash
# 1) uv 与依赖
curl -LsSf https://astral.sh/uv/install.sh | sh          # 或 brew install uv
git clone <fork>/agentic-rl-lab.git && cd agentic-rl-lab
uv sync                                                   # 按 uv.lock 建 .venv
uv sync --extra alfworld                                  # 只有 ALFWorld/TEMPO/AgentOPSD 需要

# 2) 版本自检（期望 0.2.9 0.10.4，更大也可以）
uv run python -c "from importlib.metadata import version; print(version('pytrio'), version('swanlab'))"

# 3) PyTRIO（命令是 trio，不是 pytrio）
uv run trio login            # key 在 pytrio.com/dashboard 复制，粘贴时不回显是正常的
uv run trio login -k <KEY>   # 非交互场景
export PYTRIO_API_KEY=<KEY>  # 或者用环境变量

# 4) 连通性验证（几乎不耗额度）
uv run python - <<'PY'
import pytrio as trio
c = trio.ServiceClient()
print(*c.get_supported_models(), sep="\n")
PY

# 5) SwanLab
uv run swanlab login
uv run swanlab verify
uv run swanlab ping          # 上传失败/模式回退时先查网络

# 6) 第一次训练（grpo demo，小参数）
uv run python experiments/grpo/01-demo-sync.py \
    --steps 10 --batch-size 4 --group-size 8 \
    --max-tokens 512 --swanlab-mode online
uv run swanlab watch experiments/grpo/swanlog   # http://127.0.0.1:5092
```

### 踩坑备查

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `command not found: trio/swanlab` | 可执行文件在 `.venv` 里 | 加 `uv run` 前缀，或 `source .venv/bin/activate` |
| 粘贴 key 后屏幕无反应 | 输入不回显 | 正常，直接回车 |
| `uv sync` 慢 | 网络 | `UV_DEFAULT_INDEX=https://mirrors.cernet.edu.cn/pypi/web/simple uv sync` |
| GSM8K 下载超时 | HF 直连 | `export HF_ENDPOINT=https://hf-mirror.com`（部分章节走 ModelScope，看各章 `start.md`） |
| SwanLab 上传失败 | 网络/凭据 | 先 `swanlab ping`，再 `swanlab verify` |

### SwanLab 四种模式

| 模式 | 数据去向 | 用途 |
| --- | --- | --- |
| `online` | 本地 + 云端 | 默认，需登录；**交作业/组队评审要用这个** |
| `local` | 仅本地 | 不想注册，`swanlab watch` 看 |
| `offline` | 仅本地，可 `swanlab sync` 补传 | 暂时断网 |
| `disabled` | 不记录 | 只跑训练看输出 |

各脚本支持的取值不完全一致（grpo 只有 `online`/`disabled`），以 `--help` 为准。

## 4. 概念拆解（读懂代码的最小词汇表）

**SFT vs RL。** SFT 给标准答案，按 `cross_entropy` 学；RL 让模型自己答，用 reward 打分，强化好的压差。
本 Lab 里 GRPO 之后基本都是 RL 或蒸馏，SFT 只在 Spec-o3、Medical OPD 里当热启动。

**policy / rollout / on-policy。** policy 就是当前在训的权重；从一条 prompt 出发生成一次回答叫一条 rollout。
on-policy 意味着训练数据由**当前**权重采出来——所以 demo 里每个 step 都「先存权重拿 sampler → 采样 → 更新」。
旧权重采的数据是 off-policy，新旧之间的偏差靠重要性比率补偿。

**advantage 与组（GRPO 的核心简化）。** 同一道题采 `group_size` 条回答，
`advantage = reward − 组内均值 reward`：高于均值的被鼓励，低于的被压低。
**不需要额外训 critic**——这是 GRPO 相对 PPO 最大的简化。
副产物：整组 reward 全同（全对或全错）时 advantage 全 0，这组白采，日志里叫 **degenerate group**。
→ 读 GRPO 时要盯 `group_size` 与 degenerate 比例：题太简单或太难都会浪费采样。

**old logprobs 与重要性比率。** logprob 是每个 token 的对数概率；同一 token 在新旧 policy 下的概率比
就是重要性采样比率。RL loss 基本都是「对比率乘 advantage 做文章」：`ppo` 给比率加 clip，`cispo` 换 clip 的位置。
→ 顺序建议：先读 01 Loss Functions，再读 02 GRPO。

**LoRA 与权重。** 不动 base model，只训低秩适配器（demo 里 rank 32）。训练产物两份：
`save_weights_for_sampler`（给采样/评测用）与 `save_state`（完整 checkpoint，续训用）。权重地址是 `trio://` 形式。

**OPD（On-Policy Distillation）。** 照样用 student 自己采的 rollout，但信号不是 reward：
teacher 对 student 的回答逐 token 算 logprobs，用 student 与 teacher 的差（reverse KL）当 advantage——
等于老师逐 token 批改。OPSD 再进一步：teacher 就是 step 0 的自己，只是 prompt 里多给了参考解答。
（对应 05 General OPD、09 OPSD）

**多轮与 observation mask。** Agentic RL 让模型在回答中途调工具（搜索、代码解释器、TextWorld）。
工具返回的 observation 拼进上下文给模型看，但**不参与 loss**——环境说的话不是模型的动作，不该被强化。
代码里体现为这些 token 的 logprobs 与 advantages 填 0。
（对应 11 Search-R1、12 ReTool、13 ALFWorld）

## 5. 我自己的理解 / 补充

- 「云端只负责梯度」这个切分很像 Tinker 的 Train API 思路：**权重不在本地**，本地只有 logits 之外的东西
  （数据、reward、token 级累加）。好处是没有 GPU 门槛；代价是每个 step 都有一次网络往返，
  所以 demo 的 step 数都压得很小（几道题、十几个 step），目的是**算法级复现**而不是刷指标。
- 本地 torch 的唯一职责是「自定义 loss 的 autograd」——这解释了为什么 cispo/DAPO/GSPO 这些
  「改 loss 形状」的章节必须本地算梯度，而 `ppo` 这类内置 loss 反而更省事。
- observation mask 字段填 0 的做法，本质上和「padding 不算 loss」是同一个套路：
  用 mask 控制哪些 token 对梯度有贡献。区别是前者按语义（谁说的话）分，后者按长度分。

## 6. 待解问题（下次回来补）

- [ ] `forward_backward_custom` 的接口签名到底长什么样？本地 autograd 之后梯度怎么上传？
- [ ] degenerate group 在代码里有没有被检测/丢弃？还是只体现在日志？
- [ ] `trio://` 权重地址是 s3 之类对象存储的封装，还是自有协议？续训时 `save_state` 怎么定位？
- [ ] rank 32 的 LoRA 在这几个 demo 上够用吗？章节里有没有给出 rank 的对比实验？
