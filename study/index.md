---
layout: home

hero:
  name: Agentic RL 学习笔记
  text: 逐章精读 Datawhale Agentic RL Lab
  tagline: 环境 → Loss → GRPO → OPD → Agentic（工具调用 / 多轮）
  image:
    src: /favicon.svg
    alt: logo
  actions:
    - theme: brand
      text: 开始阅读
      link: /notes/
    - theme: alt
      text: 第 00 章：环境配置
      link: /notes/00-getting-started
    - theme: alt
      text: 上游在线手册
      link: https://datawhalechina.github.io/agentic-rl-lab/

features:
  - title: 全部在云端算梯度
    details: PyTRIO 负责前向、反向、优化器与采样，本地只跑数据、reward 和训练循环——一台联网笔记本就够，不需要 GPU。
  - title: 从 Loss 读起
    details: 先看 importance sampling 与 clip 的位置，再看 GRPO 的 advantage；否则 loss 代码只能看个寂寞。
  - title: 笔记本位固定四段
    details: 一句话结论 / 机制拆解 / 动手记录 / 待解问题。只写自己的理解，不抄原文。
---

## 这个站点是什么

[datawhalechina/agentic-rl-lab](https://github.com/datawhalechina/agentic-rl-lab) 的**个人陪读笔记**：
每读完一章写一篇，记录机制理解、实跑命令、踩坑与没想通的问题。原仓库内容不在这里重复，只放索引和我的笔记。

> 域名 `arl.biochao.cc` 与个人博客 [biochao.cc](http://biochao.cc) 完全独立：一个在 GitHub Pages，
> 一个在自己的服务器上，互不影响。
