# 学习笔记 · Agentic RL Lab

本目录是我对 [datawhalechina/agentic-rl-lab](https://github.com/datawhalechina/agentic-rl-lab) 的**私人学习笔记**，
写在 `study` 分支上，**只推送到 fork 仓库**（`biociao/agentic-rl-lab`），不向原仓库提 PR。
原仓库内容（`experiments/`、`README.md` 等）保持上游原样，笔记只放在 `study/` 下，便于随时 rebase。

- 上游：`upstream` → `datawhalechina/agentic-rl-lab`（push 已禁用为 `no_push`）
- 我的 fork：`fork` → `biociao/agentic-rl-lab`（`study` 分支跟踪 `fork/study`）

## 阅读进度

按原仓库约定的阅读顺序（[experiments/README.md](../experiments/README.md)）。

| # | 章节 | 笔记 | 状态 |
| --- | --- | --- | --- |
| 00 | 环境配置与基础概念 | [00-getting-started.md](./00-getting-started.md) | ✅ 已读 |
| 01 | Loss Functions | — | ⏳ 待读（建议在读 GRPO 之前） |
| 02 | GRPO | — | ⏳ |
| 03 | Vision GRPO | — | ⏳ |
| 04 | PyTrio-Jev | — | ⏳ |
| 05 | General OPD | — | ⏳ |
| 06 | Medical OPD | — | ⏳ |
| 07 | DAPO | — | ⏳ |
| 08 | GSPO | — | ⏳ |
| 09 | OPSD | — | ⏳ |
| 10 | AgentOPSD | — | ⏳ |
| 11 | Search-R1 | — | ⏳ |
| 12 | ReTool | — | ⏳ |
| 13 | ALFWorld | — | ⏳ |
| 14 | TEMPO | — | ⏳ |
| 15 | Spec-o3 | — | ⏳ |
| 16 | Tournament-GRPO | — | ⏳ |

## 笔记约定

每篇笔记固定四段，避免抄原文，只写"我的理解"：

1. **一句话结论**：这章解决什么问题、核心机制是什么。
2. **机制拆解**：关键公式/数据结构/代码路径，用自己的话讲，标出对应文件与行号。
3. **动手记录**：实际跑的命令、参数、SwanLab 链接、异常与修复。
4. **待解问题**：没想通的地方，留到下次。

## 常用命令

```bash
# 同步上游（只拉不推）
git fetch upstream && git rebase upstream/main

# 笔记提交后推送到 fork（study 分支已设 upstream）
git push fork study
```

## 待办

- [ ] `uv sync`，验证 `pytrio 0.2.9 / swanlab 0.10.4`
- [ ] 注册并登录 PyTRIO（`uv run trio login`）、SwanLab（`uv run swanlab login && verify`）
- [ ] 跑通第一次训练：`experiments/grpo/01-demo-sync.py`（10 步小参数）
