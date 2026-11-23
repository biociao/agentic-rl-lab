"""概念二：图文 prompt 的 token 账本，必须本地算得出来。

视觉 GRPO 与纯文本 GRPO 最大的差别在 prompt 的拼装：渲染后的模板沿 `<|image_pad|>`
切成「文本 + 图片 + 文本」三个 chunk，图片只以 patch 数承诺自己会占多少个 token。
远端采样返回后，训练脚本用 `response.input_tokens == len(prompt)` 校验这笔账。

这个脚本做三件事，全部在本地完成（首次运行需要联网取 tokenizer 与 image processor，
之后走 workspace 内的缓存；不调用付费的采样/训练接口）：

1. 对真实 GeoQA 图片算 patch 数与视觉 token 数；
2. 用真实 tokenizer 统计 100 条测试样本的 prompt 长度，并核对
   `len(ModelInput) == 文本 token + 视觉 token`；
3. 造一条假的 completion，检查 `target_tokens / logprobs / advantages`
   三个数组与 `model_input` 是否严格等长——这正是训练脚本里 `observation_length` 的来源。
"""

from __future__ import annotations

import os
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
# 缓存与端点都固定在仓库内，避免写到家目录、也避开会 302 到官方站的镜像。
os.environ.setdefault("HF_HOME", str(REPO / ".cache/hf"))
os.environ.setdefault("HF_DATASETS_CACHE", str(REPO / ".cache/hf/datasets"))
os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

sys.path.insert(0, str(REPO / "experiments/vision-grpo"))

import pytrio as trio  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers import AutoImageProcessor, AutoTokenizer  # noqa: E402

import train as vision_grpo  # noqa: E402

MODEL = "Qwen/Qwen3.5-4B"


def load_test_split():
    return load_dataset(
        "parquet",
        data_files=str(REPO / "experiments/vision-grpo/datasets/test.parquet"),
        split="train",
    )


def main() -> None:
    data = load_test_split()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    image_processor = AutoImageProcessor.from_pretrained(MODEL, backend="pil")
    merge = int(image_processor.merge_size)
    print(f"模型：{MODEL}　patch={image_processor.patch_size}px　merge={merge}×{merge}\n")

    print("前 5 条测试样本的 token 账本：")
    header = f"{'id':<7}{'尺寸(w×h)':<12}{'patch':<7}{'视觉token':<10}{'文本token':<10}{'prompt':<8}{'视觉占比'}"
    print(header)
    print("-" * len(header) * 2)
    for row in data.select(range(5)):
        chunks = vision_grpo.build_prompt_chunks(
            tokenizer, image_processor, row["image"], str(row["subject"]),
            [str(choice) for choice in row["choices"]],
        )
        head, image_chunk, tail = chunks
        patches = image_chunk.expected_tokens * merge**2
        text_tokens = len(head.tokens) + len(tail.tokens)
        total = len(trio.ModelInput(chunks=chunks))
        width, height = row["image"].size
        print(
            f"{row['id']:<7}{f'{width}×{height}':<12}{patches:<7}"
            f"{image_chunk.expected_tokens:<10}{text_tokens:<10}{total:<8}"
            f"{image_chunk.expected_tokens / total:.0%}"
        )

    image_tokens: list[int] = []
    text_tokens_all: list[int] = []
    prompt_lengths: list[int] = []
    mismatches = 0
    for row in data:
        chunks = vision_grpo.build_prompt_chunks(
            tokenizer, image_processor, row["image"], str(row["subject"]),
            [str(choice) for choice in row["choices"]],
        )
        expected = len(chunks[0].tokens) + chunks[1].expected_tokens + len(chunks[2].tokens)
        actual = len(trio.ModelInput(chunks=chunks))
        mismatches += expected != actual
        image_tokens.append(chunks[1].expected_tokens)
        text_tokens_all.append(len(chunks[0].tokens) + len(chunks[2].tokens))
        prompt_lengths.append(actual)

    def summary(name: str, values: list[int]) -> str:
        return (
            f"  {name:<12}min={min(values):<5}中位={statistics.median(values):<7.1f}"
            f"max={max(values)}"
        )

    print("\n100 条测试样本：")
    print(summary("视觉 token", image_tokens))
    print(summary("文本 token", text_tokens_all))
    print(summary("prompt 总长", prompt_lengths))
    print(f"  视觉 token 占 prompt 的比例（中位）："
          f"{statistics.median(image_tokens) / statistics.median(prompt_lengths):.0%}")
    print(f"  逐条核对 len(ModelInput) == 文本+视觉：{'全部一致' if not mismatches else f'{mismatches} 条不一致'}")

    # 用真实 prompt 拼一条假 completion，看三个数组如何对齐。
    row = data[0]
    chunks = vision_grpo.build_prompt_chunks(
        tokenizer, image_processor, row["image"], str(row["subject"]),
        [str(choice) for choice in row["choices"]],
    )
    completion_length = 40
    sample = vision_grpo.RolloutSample(
        tokens=[42] * completion_length,
        logprobs=[-0.5] * completion_length,
        text="（假的 completion）",
        predicted_choice="C",
        reward=1.0,
        advantage=0.25,
    )
    group = vision_grpo.RolloutGroup(
        prompt_chunks=chunks,
        prompt_length=len(trio.ModelInput(chunks=chunks)),
        samples=[sample],
    )
    datum = vision_grpo.build_grpo_datum(group, sample)
    model_input_length = len(datum.model_input)
    # loss_fn_inputs 里的值是 TensorData，取长度要先转成 list。
    arrays = {name: len(field.tolist()) for name, field in datum.loss_fn_inputs.items()}
    observation_length = group.prompt_length - 1
    print(f"\n拼一条 {completion_length} token 的假 completion（prompt={group.prompt_length}）：")
    print(f"  model_input      = {model_input_length}  tokens"
          f"（prompt {group.prompt_length} + completion 去掉末位 {completion_length - 1}）")
    print(f"  observation_length = {observation_length}（prompt - 1，前面全是 0 占位）")
    for name, length in arrays.items():
        print(f"  {name:<15}= {length}  {'✓' if length == model_input_length else '✗ 与 model_input 不等长'}")
    print(f"  completion 区间取值："
          f"target={datum.loss_fn_inputs['target_tokens'].tolist()[-3:]} "
          f"logprobs={datum.loss_fn_inputs['logprobs'].tolist()[-2:]} "
          f"advantages={datum.loss_fn_inputs['advantages'].tolist()[-2:]}")
    if any(length != model_input_length for length in arrays.values()):
        raise SystemExit("三个数组与 model_input 不等长，策略梯度会错位")


if __name__ == "__main__":
    main()
