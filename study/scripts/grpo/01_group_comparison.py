"""概念一：同一道题答几遍，拿组内平均分当参照。

GRPO 不训练一个专门估分的模型。它让模型对同一道题回答好几遍，
再用这组回答的平均分当作"参照线"：比参照线高的答案被鼓励，比参照线低的被压低。

这个脚本不联网、不消耗额度，它直接调用训练脚本里真实的打分和优势计算函数，
喂进去的回答是固定的，所以每次跑出来的数字都一样。
"""

from pathlib import Path
import runpy
from types import SimpleNamespace


REPO = Path(__file__).resolve().parents[3]


def load_demo():
    """载入训练脚本，拿到里面真实的函数。"""
    return runpy.run_path(str(REPO / "experiments/grpo/01-demo-sync.py"))


def rollout_with_fixed_answers(demo, answers, ground_truth="12"):
    """用固定回答冒充采样结果，走一遍真实的采样、打分、算优势的流程。"""

    def fake_sample(**kwargs):
        sequences = [
            SimpleNamespace(tokens=[21, 22], logprobs=[-1.0, -2.0], text=answer)
            for answer in answers
        ]
        return SimpleNamespace(
            result=lambda: SimpleNamespace(sequences=sequences)
        )

    sampler = SimpleNamespace(sample=fake_sample)
    return demo["run_rollout_group"](
        sampler, None, [11, 12, 13], ground_truth, None, len(answers)
    )


def show(title, samples):
    rewards = [round(sample.reward, 2) for sample in samples]
    advantages = [round(sample.advantage, 2) for sample in samples]
    mean = sum(rewards) / len(rewards)
    print(f"\n{title}")
    print(f"  每条回答的得分：{rewards}")
    print(f"  这道题的平均分：{mean:.2f}")
    print(f"  减去平均分以后：{advantages}")


def main():
    demo = load_demo()

    # 三对一错：答对的比平均分高，答错的比平均分低。
    show(
        "情况一：四条回答里两条答对（标准答案是 12）",
        rollout_with_fixed_answers(
            demo, [r"\boxed{12}", r"\boxed{11}", r"\boxed{12}", "算不出来"]
        ),
    )

    # 唯一答对：对的这条领先得更多。
    show(
        "情况二：只有一条答对",
        rollout_with_fixed_answers(
            demo, [r"\boxed{12}", r"\boxed{11}", r"\boxed{13}", r"\boxed{10}"]
        ),
    )

    # 全对和全错：没有相对差别，优势全是 0。
    show(
        "情况三：四条都答对",
        rollout_with_fixed_answers(demo, [r"\boxed{12}"] * 4),
    )
    show(
        "情况四：四条都答错",
        rollout_with_fixed_answers(demo, [r"\boxed{11}"] * 4),
    )

    print("\n结论：优势衡量的是「比同组其他回答好多少」，不是「答对没有」。")
    print("      全对和全错都没有参照差，这一组不会带来任何更新。")


if __name__ == "__main__":
    main()
