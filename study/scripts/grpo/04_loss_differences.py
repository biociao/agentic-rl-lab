"""概念四：同样一条回答，三种损失各自会怎么推它。

到这里，我们已经有了两个东西：
  1. 这条回答比同组平均好多少（优势）；
  2. 现在和采样时相比，更想要还是更不想要这个词（概率比）。

剩下的问题只有一个：怎么把这两样东西变成梯度。教程里的三种损失，
差别恰好都在这一步。这个脚本用同一个词元、同一个优势值，把三种算法各求一次梯度。

注意：这是本地的小算式，用的是教程里的公式和训练脚本里真实的 CISPO 实现，
不是模型参数的梯度，也不代表真实训练时的数值大小。它只回答一个问题：
这个位置上的学习，是被推了一把、被关掉了，还是力度被限制了。
"""

import math
from pathlib import Path
import runpy

import torch


REPO = Path(__file__).resolve().parents[3]


def load_demo():
    return runpy.run_path(str(REPO / "experiments/grpo/01-demo-sync.py"))


def gradients(demo, advantage):
    """采样时概率 10%，训练时 50%，即概率比为 5。"""
    sampling_logprob = math.log(0.10)
    target_logprob = torch.tensor([math.log(0.50)], requires_grad=True)
    ratio = torch.exp(target_logprob - sampling_logprob)

    重要性采样 = -(ratio * advantage).sum()
    梯度_重要性采样 = torch.autograd.grad(
        重要性采样, target_logprob, retain_graph=True
    )[0].item()

    ppo = -torch.minimum(
        ratio * advantage, ratio.clamp(0.8, 1.2) * advantage
    ).sum()
    梯度_ppo = torch.autograd.grad(ppo, target_logprob, retain_graph=True)[0].item()

    # CISPO 用训练脚本里真实的那份实现（裁剪后 detach，只留 logprob 走梯度）。
    cispo_fn = demo["make_cispo_loss_fn"](
        [[sampling_logprob]], [[advantage]], 0.0, 4.0
    )
    cispo, _ = cispo_fn([None], [target_logprob])
    梯度_cispo = torch.autograd.grad(cispo, target_logprob)[0].item()

    return 梯度_重要性采样, 梯度_ppo, 梯度_cispo


def main():
    demo = load_demo()

    for advantage, 标题 in (
        (0.5, "优势为正：这条回答值得鼓励"),
        (-0.5, "优势为负：这条回答应该被压低"),
    ):
        a, b, c = gradients(demo, advantage)
        print(f"\n{标题}（优势 = {advantage:+.1f}，概率比 = 5）")
        print(f"  重要性采样：梯度 {a:+.2f}")

        if abs(b) < 1e-9:
            print("  PPO：       梯度  0.00  ← 这一项被裁剪掉了，不再继续推")
        else:
            print(f"  PPO：       梯度 {b:+.2f}")

        print(f"  CISPO：     梯度 {c:+.2f}")

    print("\n怎么读这组数字：")
    print("  重要性采样：老老实实按概率比乘优势，不留余地，所以推得最狠。")
    print("  PPO：比值冲到上界时，正优势的推会被叫停（梯度变 0）；")
    print("       负优势时它不会停，因为「压得过低」并不在裁剪的范围内。")
    print("  CISPO：同样限制力度，但限制的是乘上去的权重，")
    print("         「继续学」这条路没有被堵死，所以梯度还在，只是变小了。")


if __name__ == "__main__":
    main()
