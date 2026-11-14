# 学习路线与进度

这里记录我对 [datawhalechina/agentic-rl-lab](https://github.com/datawhalechina/agentic-rl-lab) 的陪读进度。
**已读**的章节链到本地笔记，**未读**的链到上游原文——点开就能对照着看。

> 笔记写在仓库 `study` 分支的 `study/notes/` 目录，只推送到我的 fork，不向上游提 PR。

## 进度

| # | 章节 | 笔记 / 原文 | 状态 |
| --- | --- | --- | --- |
| 00 | 环境配置与基础概念 | [笔记](./00-getting-started.md) · [原文](https://datawhalechina.github.io/agentic-rl-lab/experiments/getting-started/) | ✅ 已读 |
| 01 | Loss Functions | [原文](https://datawhalechina.github.io/agentic-rl-lab/experiments/loss-functions/) | ⏳ 待读（读 GRPO 前必读） |
| 02 | GRPO | [原文](https://datawhalechina.github.io/agentic-rl-lab/experiments/grpo/) | ⏳ |
| 03 | Vision GRPO | [原文](https://datawhalechina.github.io/agentic-rl-lab/experiments/vision-grpo/) | ⏳ |
| 04 | PyTrio-Jev | [原文](https://datawhalechina.github.io/agentic-rl-lab/experiments/pytrio-jev/) | ⏳ |
| 05 | General OPD | [原文](https://datawhalechina.github.io/agentic-rl-lab/experiments/general-opd/) | ⏳ |
| 06 | Medical OPD | [原文](https://datawhalechina.github.io/agentic-rl-lab/experiments/medical-opd/) | ⏳ |
| 07 | DAPO | [原文](https://datawhalechina.github.io/agentic-rl-lab/experiments/dapo/) | ⏳ |
| 08 | GSPO | [原文](https://datawhalechina.github.io/agentic-rl-lab/experiments/gspo/) | ⏳ |
| 09 | OPSD | [原文](https://datawhalechina.github.io/agentic-rl-lab/experiments/opsd/) | ⏳ |
| 10 | AgentOPSD | [原文](https://datawhalechina.github.io/agentic-rl-lab/experiments/agentopsd/) | ⏳ |
| 11 | Search-R1 | [原文](https://datawhalechina.github.io/agentic-rl-lab/experiments/search-r1/) | ⏳ |
| 12 | ReTool | [原文](https://datawhalechina.github.io/agentic-rl-lab/experiments/retool/) | ⏳ |
| 13 | ALFWorld | [原文](https://datawhalechina.github.io/agentic-rl-lab/experiments/alfworld/) | ⏳ |
| 14 | TEMPO | [原文](https://datawhalechina.github.io/agentic-rl-lab/experiments/tempo/) | ⏳ |
| 15 | Spec-o3 | [原文](https://datawhalechina.github.io/agentic-rl-lab/experiments/spec-o3/) | ⏳ |
| 16 | Tournament-GRPO | [原文](https://datawhalechina.github.io/agentic-rl-lab/experiments/tournament-grpo/) | ⏳ |

## 笔记约定

每篇固定四段，避免抄原文，只写自己的理解：

1. **一句话结论**：这章解决什么问题、核心机制是什么。
2. **机制拆解**：关键公式 / 数据结构 / 代码路径，用自己的话讲，尽量标出对应文件。
3. **动手记录**：实际跑的命令、参数、SwanLab 链接、异常与修复。
4. **待解问题**：没想通的地方，留到下次。

## 待办

- [ ] `uv sync`，验证 `pytrio 0.2.9 / swanlab 0.10.4`
- [ ] 注册并登录 PyTRIO（`uv run trio login`）、SwanLab（`uv run swanlab login && verify`）
- [ ] 跑通第一次训练：`experiments/grpo/01-demo-sync.py`
- [ ] 读 01 Loss Functions，补一篇笔记
