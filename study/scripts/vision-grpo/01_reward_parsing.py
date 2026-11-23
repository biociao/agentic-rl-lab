"""概念一：规则奖励只看最后一个 \\boxed{} 选项。

这个脚本不联网、不消耗额度。它直接调用 `experiments/vision-grpo/train.py` 里真实的
正则与解析函数，喂进去的回答是固定的，所以每次跑出来的结果都一样。

要点：
- 只认 A–D 单个字母，`\\boxed{AB}` 不算；
- 出现多个 boxed 时取最后一个（容忍推理中途的临时候选）；
- 没有 boxed、或被截断在半截，都得 0 分。
"""

from __future__ import annotations

import runpy
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

CASES = [
    ("标准写法", r"由勾股定理得 AC=5，故选 \boxed{C}", "C"),
    ("空格 + 小写", r"所以答案是 \boxed { b }", "B"),
    ("推理中途改口", r"先猜 \boxed{A}，重新算一遍后改选 \boxed{D}", "D"),
    ("结尾重复一次", r"\boxed{C}。理由同上，最终 \boxed{C}", "C"),
    ("两个字母", r"\boxed{AB}", None),
    ("没给 boxed（数据集原写法）", r"解：……∴线段 AD 的最小值为 1，故选:A．", None),
    ("截断在半截（1024 token 上限）", r"设圆心为 E，连接 BE、CE，则 \boxed{", None),
    ("选项正确但写错字母", r"答案显然是 \boxed{B}", "B"),
]


def main() -> None:
    train = runpy.run_path(str(REPO / "experiments/vision-grpo/train.py"))
    extract_choice = train["extract_choice"]

    print("解析函数：train.extract_choice")
    print(f"正则：{train['BOXED_CHOICE_PATTERN'].pattern}\n")
    print(f"{'情形':<26}{'回答':<44}{'解析结果':<10}{'判定'}")
    print("-" * 96)

    failures = 0
    for name, text, expected in CASES:
        got = extract_choice(text)
        ok = got == expected
        failures += not ok
        shown = text if len(text) <= 40 else text[:37] + "…"
        print(f"{name:<26}{shown:<44}{str(got):<10}{'✓' if ok else '✗ 与预期不符'}")

    # 再走一遍组内优势：这 8 条回答里，只有 3 条与标准答案 C 相符。
    gold = "C"
    rewards = [1.0 if extract_choice(text) == gold else 0.0 for _, text, _ in CASES]
    mean = sum(rewards) / len(rewards)
    advantages = [reward - mean for reward in rewards]
    print(f"\n标准答案为 C 时这一组的得分：{rewards}")
    print(f"组内平均分：{mean:.3f}")
    print(f"逐条优势：{[round(a, 3) for a in advantages]}")
    print(f"进入 backward 的回答数（优势非零）：{sum(a != 0 for a in advantages)}")

    if failures:
        raise SystemExit(f"{failures} 例与预期不符")


if __name__ == "__main__":
    main()
