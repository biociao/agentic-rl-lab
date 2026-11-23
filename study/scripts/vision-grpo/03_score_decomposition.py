"""概念三：把 +16pp 的涨点拆成「格式」和「能力」两份。

教程报告的 100 条固定测试集结果：

    Base              准确率 71.0%　格式率 75.0%
    GRPO step 100     准确率 87.0%　格式率 91.0%

因为「答对」必须建立在「能解析出 \\boxed{}」之上，准确率的分子不可能超过格式率的分子。
把两者相乘（准确率 = 格式率 × 条件准确率）就能看出涨点到底花在哪里。

这个脚本不联网、不消耗额度，只做算术与一致性检查；标签分布读本地 test.parquet。
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
os.environ.setdefault("HF_HOME", str(REPO / ".cache/hf"))
os.environ.setdefault("HF_DATASETS_CACHE", str(REPO / ".cache/hf/datasets"))

TOTAL = 100
BASE_ACCURATE, BASE_FORMATTED = 71, 75
STEP_ACCURATE, STEP_FORMATTED = 87, 91


def main() -> None:
    base_cond = BASE_ACCURATE / BASE_FORMATTED
    step_cond = STEP_ACCURATE / STEP_FORMATTED
    base_acc = BASE_ACCURATE / TOTAL
    step_acc = STEP_ACCURATE / TOTAL
    format_gain = (STEP_FORMATTED - BASE_FORMATTED) / TOTAL
    cond_gain = step_cond - base_cond
    total_gain = step_acc - base_acc

    print("一、两组数字")
    print(f"  Base        准确 {BASE_ACCURATE}/100 = {base_acc:.1%}　"
          f"格式 {BASE_FORMATTED}/100 = {BASE_FORMATTED:.1f}%　"
          f"条件准确率 {BASE_ACCURATE}/{BASE_FORMATTED} = {base_cond:.1%}")
    print(f"  step 100    准确 {STEP_ACCURATE}/100 = {step_acc:.1%}　"
          f"格式 {STEP_FORMATTED}/100 = {STEP_FORMATTED:.1f}%　"
          f"条件准确率 {STEP_ACCURATE}/{STEP_FORMATTED} = {step_cond:.1%}")

    print("\n二、一致性检查：多出来的 16 条是谁")
    newly_formatted = STEP_FORMATTED - BASE_FORMATTED
    newly_correct = STEP_ACCURATE - BASE_ACCURATE
    print(f"  新变得可解析的回答：{STEP_FORMATTED} − {BASE_FORMATTED} = {newly_formatted} 条")
    print(f"  新增答对的回答：    {STEP_ACCURATE} − {BASE_ACCURATE} = {newly_correct} 条")
    print(f"  ⇒ 这 {newly_formatted} 条全都答对了；原先能解析且答对的 {BASE_ACCURATE} 条一条没丢。")
    assert newly_formatted == newly_correct

    print("\n三、拆账（中点分解）")
    # 准确率 = f × c；Δ = Δf·c̄ + Δc·f̄，两项之和严格等于总涨点。
    f_mean = (BASE_FORMATTED + STEP_FORMATTED) / 2 / TOTAL
    c_mean = (base_cond + step_cond) / 2
    from_format = format_gain * c_mean
    from_reasoning = cond_gain * f_mean
    print(f"  格式变化贡献：{format_gain:+.1%} 格式率 × {c_mean:.1%} 平均条件准确率 = {from_format:+.2%}")
    print(f"  能力变化贡献：{cond_gain:+.2%} 条件准确率 × {f_mean:.0%} 平均格式率 = {from_reasoning:+.2%}")
    print(f"  合计：{from_format + from_reasoning:+.2%}（总涨点 {total_gain:+.2%}）")
    assert abs(from_format + from_reasoning - total_gain) < 1e-9

    print("\n四、两条硬约束")
    print(f"  1) 准确率 ≤ 格式率：Base 的格式率只有 {BASE_FORMATTED}%，"
          f"所以哪怕条件准确率做到 100%，也到不了 {step_acc:.0%}——")
    print(f"     想拿 {step_acc:.0%} 的准确率，格式率至少要先做到 {step_acc:.0%}。")
    print(f"  2) 若格式率停在 {BASE_FORMATTED}% 不动，条件准确率需要 "
          f"{step_acc * TOTAL / BASE_FORMATTED:.0%} 才能达到同样的准确率——已经超过 100%，不可能。")
    assert step_acc * TOTAL / BASE_FORMATTED > 1

    print("\n五、这 100 条题的标签先验（本地 test.parquet）")
    from datasets import load_dataset

    test = load_dataset(
        "parquet",
        data_files=str(REPO / "experiments/vision-grpo/datasets/test.parquet"),
        split="train",
    )
    counts = Counter(int(label) for label in test["label"])
    letters = "ABCD"
    shares = {letters[index]: counts.get(index, 0) for index in range(4)}
    best_letter, best_count = max(shares.items(), key=lambda item: item[1])
    print("  " + "　".join(f"{letter}={count}" for letter, count in shares.items()))
    print(f"  「永远选 {best_letter}」的准确率 = {best_count}%　"
          f"随机猜 = {TOTAL // 4}%　Base = {BASE_ACCURATE}%")


if __name__ == "__main__":
    main()
