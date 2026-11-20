"""概念二：一条回答的得分，怎么变成每个词元（token）上的训练信号。

模型生成回答是一步一步来的，但打分只在最后做一次。
那么"这条回答不错"这个结论，是怎么落到每个词元上的？

这份训练脚本的做法是：把整条回答的优势值，复制给它生成的每一个词元；
题目（prompt）部分的优势填 0，只当作上下文，不参与训练。

这个脚本调用训练脚本里真实的组装函数，把一条回答变成训练样本，
再把里面的数字打出来对照，不联网、不消耗额度。
"""

from pathlib import Path
import runpy


REPO = Path(__file__).resolve().parents[3]


def load_demo():
    return runpy.run_path(str(REPO / "experiments/grpo/01-demo-sync.py"))


def show(demo, prompt_tokens, answer_tokens, advantage=0.5):
    sample = demo["RolloutSample"](
        tokens=answer_tokens,
        logprobs=[-1.0] * len(answer_tokens),
        text="（这里只是占位，不打分）",
        reward=1.0,
        advantage=advantage,
    )
    datum = demo["build_grpo_datum"](prompt_tokens, sample)

    inputs = datum.model_input.chunks[0].tokens
    targets = datum.loss_fn_inputs["target_tokens"].data
    advantages = datum.loss_fn_inputs["advantages"].data
    trainable = sum(1 for value in advantages if float(value) != 0.0)

    print(f"\n题目 {len(prompt_tokens)} 个词元，回答 {len(answer_tokens)} 个词元")
    print(f"  喂给模型的输入： {inputs}")
    print(f"  要它预测的下一个词元： {targets}")
    print(f"  每个位置的优势：     {[round(float(v), 2) for v in advantages]}")
    print(f"  真正参与训练的词元： {trainable} 个（其余是题目部分，填 0）")


def main():
    demo = load_demo()

    show(demo, prompt_tokens=[11, 12, 13], answer_tokens=[21, 22])

    print("\n换个更长的题目再看一眼：题目变长，被填 0 的位置跟着变多。")
    show(demo, prompt_tokens=[11, 12, 13, 14, 15], answer_tokens=[21, 22])

    print("\n注意两点：")
    print("  1. 输入比预测目标多出来的那个位置，是回答的最后一个词元在预测结束。")
    print("  2. 题目部分填 0，不是因为是坏数据，而是因为那不是模型要说的话。")


if __name__ == "__main__":
    main()
