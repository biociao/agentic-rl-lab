#!/usr/bin/env python3
"""Rebuild SVGs: python3 study/scripts/draw_getting_started.py (stdlib only).

Edit labels/layout here for reproducible changes; rebuilding replaces the SVGs.
"""
from html import escape
from pathlib import Path
import re
from unicodedata import east_asian_width

OUT = Path(__file__).resolve().parents[1] / "notes/assets/00-getting-started"
INK, MUTED = "#17334a", "#486174"
BLUE, CYAN, PURPLE, AMBER = "#2265a5", "#087e8b", "#7550a5", "#99600c"
PALE = {BLUE: "#edf4fc", CYAN: "#e9f7f8", PURPLE: "#f4effa", AMBER: "#fff5e3"}


class Diagram:
    def __init__(self, name, title, subtitle, height):
        self.name, self.height = name, height
        self.parts = [f'''<svg xmlns="http://www.w3.org/2000/svg" width="840" height="{height}" viewBox="0 0 840 {height}" role="img" aria-labelledby="title desc">
<title id="title">{escape(title)}</title><desc id="desc">{escape(subtitle)}</desc>
<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto-start-reverse"><path d="M0 0 L9 4.5 L0 9 Z" fill="#486174"/></marker></defs>
<style>text {{ font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC", sans-serif; fill: {INK}; }} .title {{ font-size: 25px; font-weight: 700; }} .sub {{ font-size: 17px; fill: {MUTED}; }} .head {{ font-size: 20px; font-weight: 650; }} .body {{ font-size: 18px; }} .small {{ font-size: 16px; fill: {MUTED}; }}</style>
<rect width="840" height="{height}" rx="16" fill="#fff"/>
<rect x="1" y="1" width="838" height="{height-2}" rx="16" fill="none" stroke="#d8e3ed"/>
<rect x="28" y="28" width="5" height="56" rx="2" fill="{CYAN}"/>''']
        self.text(48, 53, title, "title")
        self.text(48, 81, subtitle, "sub")

    def text(self, x, y, text, cls="body", anchor="start"):
        self.parts.append(f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{escape(text)}</text>')

    def box(self, x, y, w, h, title, lines=(), color=BLUE):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{PALE[color]}" stroke="{color}" stroke-width="1.5"/>')
        def wrap(text, size):
            result, row, width = [], "", 0
            for char in re.findall(r"[A-Za-z_][A-Za-z_0-9]*|[^A-Za-z_]", text):
                if char == "\n":
                    result.append(row.rstrip())
                    row, width = "", 0
                    continue
                advance = sum(size * (1 if east_asian_width(c) in "WF" else 0.58) for c in char)
                if row and width + advance > w - 36:
                    result.append(row.rstrip())
                    row, width = "", 0
                if not row and char == " ":
                    continue
                row += char
                width += advance
            return result + [row]
        headings = wrap(title, 20)
        for i, row in enumerate(headings):
            self.text(x+18, y+31+i*26, row, "head")
        rows = [row for line in lines for row in wrap(line, 18)]
        for i, row in enumerate(rows):
            self.text(x+18, y+61+(len(headings)-1)*26+i*27, row)
        last_baseline = 61+(len(headings)-1)*26+(len(rows)-1)*27 if rows else 31+(len(headings)-1)*26
        if last_baseline > h-12:
            raise ValueError(f"{self.name}: card too short for {title!r}")

    def arrow(self, points, label=None, label_pos=None, dashed=False):
        path = "M" + " L".join(f"{x},{y}" for x, y in points)
        dash = ' stroke-dasharray="6 5"' if dashed else ""
        self.parts.append(f'<path d="{path}" fill="none" stroke="{MUTED}" stroke-width="2" marker-end="url(#arrow)"{dash}/>')
        if label:
            self.text(*label_pos, label, "small", "middle")

    def note(self, y, lines, color=CYAN):
        h = 25*len(lines)+24
        self.parts.append(f'<rect x="28" y="{y}" width="784" height="{h}" rx="10" fill="{PALE[color]}"/>')
        for i, line in enumerate(lines):
            self.text(44, y+28+i*25, line, "sub")

    def save(self):
        self.parts.append('</svg>')
        (OUT / f"{self.name}.svg").write_text("\n".join(self.parts)+"\n", encoding="utf-8")


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    d = Diagram("01-platforms", "平台全景：谁负责什么？", "本地编排实验，远程提供算力，日志保留实验轨迹。", 478)
    d.box(28, 120, 300, 138, "本地 Python 脚本", ["数据、reward 与 loss", "CPU 编排训练循环"], CYAN)
    d.box(420, 120, 392, 138, "PyTRIO · 远程 GPU", ["采样 / 前向 / 参数反向传播", "优化器更新 / 保存权重"], BLUE)
    d.arrow([(328, 160), (420, 160)], "SDK 请求", (374, 145))
    d.arrow([(420, 220), (328, 220)], "结果返回", (374, 246))
    d.box(28, 310, 300, 130, "uv + uv.lock", ["固定依赖与环境", "让脚本能被重复运行"], PURPLE)
    d.box(420, 310, 392, 130, "SwanLab", ["记录配置、指标和实验对比", "观察训练是否朝目标前进"], AMBER)
    d.arrow([(178, 310), (178, 258)])
    d.arrow([(328, 258), (374, 283), (420, 310)])
    d.save()

    d = Diagram("02-pytrio", "PyTRIO：两条 loss 执行路径", "参数反向传播与优化器更新始终由服务端完成。", 642)
    d.box(28, 118, 340, 126, "本地 · 组织 Datum", ["model_input：输入 token", "loss_fn_inputs：目标与权重"], CYAN)
    d.box(464, 118, 348, 126, "云端 · 模型前向", ["模型权重 → 逐 token logprob", "内置 loss 也在云端计算"], BLUE)
    d.arrow([(368, 181), (464, 181)], "提交样本", (416, 164))
    d.box(28, 302, 340, 126, "本地 · 自定义 loss", ["CPU torch：logprob → 标量 L", "求 ∂L/∂logprob"], CYAN)
    d.box(464, 302, 348, 126, "云端 · 参数反向传播", ["内置 loss：直接得到梯度", "自定义：接收 logprob 的梯度"], BLUE)
    d.arrow([(464, 229), (416, 229), (416, 275), (198, 275), (198, 302)], "自定义路径：取回 logprob", (196, 269))
    d.arrow([(368, 365), (464, 365)], "回传梯度", (416, 350))
    d.arrow([(700, 244), (700, 302)], "内置路径", (750, 278))
    d.box(464, 474, 348, 126, "optim_step → 更新 LoRA", ["save_weights_for_sampler", "trio:// 快照 → sampler"], BLUE)
    d.arrow([(638, 428), (638, 474)])
    d.box(28, 474, 340, 126, "为什么 CPU 就够？", ["本地只求小图的梯度", "大模型权重的计算在云端"], CYAN)
    d.save()

    d = Diagram("03-monitoring", "RL 监控：reward 上升之后，还要看什么？", "下图是诊断逻辑示意，不代表任何真实实验结果。", 516)
    d.box(245, 116, 350, 88, "观察到 reward 均值上升", ["先保留结论，继续联合检查"], CYAN)
    for x, title, lines in [(28, "回答长度", ["是否不断变长？", "是否频繁触及上限？"]), (296, "有效学习信号", ["degenerate_fraction", "组内是否还有差异？"]), (564, "策略更新状态", ["clip fraction / entropy", "更新是否过猛、探索是否减少？"])]:
        if x == 564:
            lines = ["clip fraction", "entropy", "更新幅度与探索程度"]
        d.box(x, 266, 248, 128, title, lines, BLUE)
        d.arrow([(420, 204), (420, 231), (x+124, 231), (x+124, 266)])
    d.note(421, ["把 reward、长度、退化组比例与策略指标放在同一实验时间轴上。", "再结合任务评测判断：得分改善是否对应能力改善。"], AMBER)
    d.save()

    d = Diagram("04-environment", "环境也是实验输入", "锁定依赖，是复现的起点；数据和运行条件也需要记录。", 477)
    d.box(28, 121, 340, 120, "环境约束", ["pyproject.toml · Python 版本", "uv.lock · 精确依赖版本"], PURPLE)
    d.box(464, 121, 348, 120, "uv sync", ["按项目约束同步环境", "运行同一份训练脚本"], CYAN)
    d.arrow([(368, 181), (464, 181)])
    d.box(28, 293, 340, 127, "实验条件", ["数据版本 · tokenizer / 模板", "seed · 模型 · 超参数"], BLUE)
    d.box(464, 293, 348, 127, "可追溯的一次实验", ["保存环境标识与运行配置", "与指标记录一起比较"], AMBER)
    d.arrow([(638, 241), (638, 293)])
    d.arrow([(368, 356), (464, 356)])
    d.save()

    d = Diagram("05-concept-map", "概念地图：从目标到一次更新", "主线是采样—反馈—更新；蒸馏改变反馈来源，Agentic 扩展轨迹。", 546)
    d.box(28, 120, 230, 115, "① policy 采样", ["prompt → rollout", "当前策略生成回答"], CYAN)
    d.box(305, 120, 230, 115, "② 评价回答", ["reward / 老师反馈", "确定学习信号"], AMBER)
    d.box(582, 120, 230, 115, "③ 组织更新", ["advantage + loss", "得到参数梯度"], BLUE)
    d.arrow([(258, 177), (305, 177)])
    d.arrow([(535, 177), (582, 177)])
    d.arrow([(697, 235), (697, 269), (143, 269), (143, 235)], "更新 policy，再采样", (420, 263))
    d.box(28, 319, 370, 124, "蒸馏支线 · OPD / OPSD", ["学生仍生成自己的轨迹", "老师提供逐 token 分布反馈"], PURPLE)
    d.box(442, 319, 370, 124, "Agentic 扩展 · 多轮交互", ["动作 → 工具 / 环境 → observation", "mask 区分谁生成了哪些 token"], CYAN)
    d.note(465, ["GRPO 决定组内如何形成 advantage；IS / PPO / CISPO / GSPO 处理更新目标。"])
    d.save()

    d = Diagram("06-supervision", "PT、SFT、RL：学习信号从哪里来？", "这里对照的是本课的典型设置；RL 也可以使用过程奖励。", 520)
    d.box(28, 120, 244, 247, "PT · 预训练", ["输入：连续文本", "目标：下一个 token", "", "[词₁] → 词₂", "[词₁ 词₂] → 词₃", "每个位置有预测目标"], BLUE)
    d.box(298, 120, 244, 247, "SFT · 监督微调", ["输入：问题 + 示范答案", "目标：示范答案 token", "", "问题 → 示范答案", "答₁ · 答₂ · 答₃", "逐 token 对照示范"], CYAN)
    d.box(568, 120, 244, 247, "RL · 强化学习", ["输入：问题", "目标：提高任务回报", "", "问题 → 自己生成回答", "回答 → reward", "再构造更新信号"], AMBER)
    d.arrow([(690, 367), (690, 406), (420, 406)])
    d.note(425, ["本课结果奖励的难点：自己采样，并把回答级反馈分配到模型的动作。", "这引出了 rollout、advantage 与 credit assignment。"], AMBER)
    d.save()

    d = Diagram("07-on-policy", "On-policy：快照、采样与更新的时间关系", "统一记号：p_old 是采样策略，p_new 是待优化策略。", 577)
    d.box(28, 120, 230, 120, "当前权重 θ_t", ["保存采样快照", "得到 sampler"], CYAN)
    d.box(305, 120, 230, 120, "由 p_old 采样", ["生成 rollout", "保留 old logprobs"], BLUE)
    d.box(582, 120, 230, 120, "用 p_new 算 loss", ["对同一批 token", "重新计算 logprobs"], PURPLE)
    d.arrow([(258, 180), (305, 180)])
    d.arrow([(535, 180), (582, 180)])
    d.box(162, 298, 516, 114, "重要性比率：目标概率 / 采样概率", ["ratio = p_new(x) / p_old(x)", "E_{p_new}[f] = E_{p_old}[ratio · f]"], BLUE)
    d.arrow([(697, 240), (697, 355), (678, 355)])
    d.arrow([(162, 355), (143, 355), (143, 240)], "更新后同步下一份快照", (420, 444))
    d.note(473, ["若快照、推理设置与参数完全一致，首次更新前 ratio 可以等于 1。", "参数更新或存在采样滞后后，比率可能偏离 1；差异越大，估计越不稳定。"], AMBER)
    d.save()

    d = Diagram("08-reward", "Reward：真实目标如何变成可计算的分数？", "奖励是目标的代理；拿到高分，需要再核对是否真的完成任务。", 530)
    d.box(222, 117, 396, 91, "真实任务目标", ["答对题 / 正确使用工具 / 完成环境任务"], CYAN)
    for x, title, lines in [(28, "可验证答案", ["最终答案是否匹配", "例：数学题答案校验"]), (296, "过程 / 格式", ["结构、步骤或工具行为", "例：格式分与搜索行为分"]), (564, "环境判定", ["任务状态是否达成", "例：ALFWorld 完成任务"])]:
        d.box(x, 267, 248, 118, title, lines, BLUE)
        d.arrow([(420, 208), (420, 235), (x+124, 235), (x+124, 267)])
    d.note(420, ["代理与目标出现偏差 → 模型可能学会利用评分规则 → reward hacking。", "核对：分数升高时，正确率、行为质量和任务完成率是否同步改善？"], AMBER)
    d.save()

    d = Diagram("09-grpo", "GRPO：同题内比较，构造 advantage", "示例只演示减均值版本；原版还使用组内标准差归一化。", 546)
    d.box(28, 117, 784, 85, "同一道题 → 采样 K = 4 条回答", ["r = [1, 1, 0, 0]      组内均值 = 0.5"], CYAN)
    d.arrow([(214, 202), (214, 242)])
    d.box(28, 242, 372, 142, "有差异的组", ["A = r − mean(r)", "A = [+0.5, +0.5, −0.5, −0.5]", "相对更好的回答被鼓励"], BLUE)
    d.box(440, 242, 372, 142, "退化组 · 所有 reward 相同", ["[1, 1, 1, 1] 或 [0, 0, 0, 0]", "减均值后 A = [0, 0, 0, 0]", "组内相对奖励没有学习信号"], AMBER)
    d.note(415, ["不另训 critic → 用组内奖励统计构造 baseline。", "代价转向数据：题目太易或太难 → 退化组增多 → 有效信号减少。", "原版归一化需处理 std = 0；DAPO 用动态采样过滤并补齐有效组。"], CYAN)
    d.save()

    d = Diagram("10-loss-family", "Loss 分支：分别改了哪一部分？", "这是改动位置的关系图，不是严格的继承链，也不是效果排名。", 708)
    d.box(222, 116, 396, 87, "共同起点：ratio × advantage", ["用采样数据估计策略更新方向"], CYAN)
    d.box(28, 263, 372, 120, "PPO · 裁剪代理目标", ["min(ratio·A, clip(ratio)·A)", "限制部分过大的更新激励"], BLUE)
    d.box(440, 263, 372, 120, "GSPO · 改变比率的粒度", ["长度归一化的序列级比率", "整条回答共享比率与裁剪"], PURPLE)
    d.arrow([(420, 203), (420, 229), (214, 229), (214, 263)])
    d.arrow([(420, 229), (626, 229), (626, 263)])
    d.box(28, 441, 372, 144, "CISPO · 改变梯度路径", ["c = detach(clip(ratio))", "c · logprob · A", "梯度仍可经 logprob 流动"], CYAN)
    d.arrow([(214, 383), (214, 441)], "处理裁剪的梯度影响", (214, 418))
    d.box(440, 441, 372, 144, "DAPO · 一组协同改动", ["Clip-Higher → 维持探索", "动态采样 → 增加有效组", "token 级 loss + 超长奖励处理"], AMBER)
    d.arrow([(400, 323), (420, 323), (420, 412), (626, 412), (626, 441)])
    d.note(618, ["读算法时依次问：采样单位是什么？ratio 如何定义？在哪里 clip？梯度从哪里走？"])
    d.save()

    d = Diagram("11-distillation", "蒸馏：谁生成轨迹，谁提供反馈？", "OPD 的关键是学生自己采样；老师对这些 token 评分，无须重新生成。", 620)
    d.box(28, 118, 244, 223, "传统离线蒸馏", ["老师 / 固定数据", "↓", "示范答案", "↓", "学生拟合示范"], BLUE)
    d.box(298, 118, 244, 223, "OPD", ["学生自己生成回答", "↓", "老师对这些 token 评分", "↓", "学生与老师的分布对比"], CYAN)
    d.box(568, 118, 244, 223, "OPSD", ["学生自己生成回答", "↓", "冻结的 step 0 自己", "额外看到参考答案", "↓ 给学生轨迹反馈"], PURPLE)
    d.box(28, 395, 372, 112, "Forward KL · 直觉", ["KL(老师 ‖ 学生)", "倾向覆盖老师有概率的模式"], BLUE)
    d.box(440, 395, 372, 112, "Reverse KL · 直觉", ["KL(学生 ‖ 老师)", "倾向集中在老师认可的模式"], CYAN)
    d.note(535, ["KL 的直觉描述的是分布偏好；实际生成行为仍受模型容量与训练设置影响。"], AMBER)
    d.save()

    d = Diagram("12-observation-mask", "Agentic RL：看得见，不代表要学着生成", "工具返回加入上下文；优化策略时，只让模型动作贡献训练信号。", 592)
    columns = [(28, 170, "prompt", "用户 / 任务", "mask = 0", BLUE), (211, 190, "模型动作 1", "发起工具调用", "mask = 1", CYAN), (414, 202, "observation", "工具 / 环境返回", "mask = 0", AMBER), (629, 183, "模型动作 2", "继续 / 最终回答", "mask = 1", CYAN)]
    for x, w, title, who, mask, color in columns:
        d.box(x, 132, w, 135, title, [who, mask], color)
    for (x, w, *_), nxt in zip(columns, columns[1:]):
        d.arrow([(x+w, 198), (nxt[0], 198)])
    d.box(28, 320, 372, 123, "上下文路径", ["prompt + 动作 + observation", "全部可供后续 token 预测使用"], BLUE)
    d.box(440, 320, 372, 123, "Loss 路径", ["Σ mask_t × token_loss_t", "模型动作保留，环境 token 屏蔽"], CYAN)
    d.arrow([(515, 267), (515, 291), (214, 291), (214, 320)])
    d.arrow([(714, 267), (714, 320)])
    d.note(476, ["掩码是逻辑含义；实现可能用 weights，或把相应 advantages 置 0。", "关键是把这些位置对 loss 的贡献置零，而不只是把 logprob 的数值写成 0。"], AMBER)
    d.save()

    d = Diagram("13-training-system", "一次训练 step：本地与云端如何接力？", "数字对应正文六个动作；虚线是更新后的下一步循环。", 713)
    d.text(220, 125, "本地 CPU · 编排与评价", "head", "middle")
    d.text(620, 125, "云端 GPU · 模型计算", "head", "middle")
    d.box(28, 153, 366, 104, "① 取 batch、拼 prompt", ["tokenizer + chat template"], CYAN)
    d.box(446, 153, 366, 104, "② 同步当前权重并采样", ["sampler → K 条回答 + logprobs"], BLUE)
    d.arrow([(394, 205), (446, 205)])
    d.box(28, 314, 366, 131, "③ 评价并打包 Datum", ["reward → 组内 advantage", "选择参与训练的 token"], CYAN)
    d.arrow([(629, 257), (629, 282), (211, 282), (211, 314)], "返回生成结果", (420, 276))
    d.box(446, 314, 366, 131, "④ forward / backward", ["内置 loss 在云端计算", "自定义 loss 经本地求导后回传"], BLUE)
    d.arrow([(394, 379), (446, 379)])
    d.box(446, 502, 366, 104, "⑤ optim_step", ["更新 LoRA 参数"], BLUE)
    d.arrow([(629, 445), (629, 502)])
    d.box(28, 502, 366, 104, "⑥ 记录指标", ["SwanLab → 比较下一步的变化"], CYAN)
    d.arrow([(446, 554), (394, 554)])
    d.arrow([(28, 554), (14, 554), (14, 205), (28, 205)], dashed=True)
    d.note(632, ["规模约束：网络往返延迟 + K × 回答长度的采样成本 + 版本与配置可追溯性。"], AMBER)
    d.save()

    d = Diagram("14-scope", "适用边界：从能跑通到能比较", "区分教学实验能说明什么，以及推广结论前还需要检查什么。", 546)
    rows = [(120, "本地无需 GPU", ["算力由远程服务提供"], "仍需核算成本", ["采样、训练、网络与配额"]), (251, "训练 LoRA 适配器", ["降低更新和权重同步的负担"], "仍有表达能力假设", ["全参训练问题需要另行评估"]), (382, "同栈内做算法对照", ["固定模型、数据与预算口径"], "跨平台需重新核对", ["实现差异、吞吐与评测设置"])]
    for y, left, ll, right, rl in rows:
        d.box(28, y, 340, 101, left, ll, CYAN)
        d.box(464, y, 348, 101, right, rl, AMBER)
        d.arrow([(368, y+50), (464, y+50)], "边界", (416, y+37))
    d.save()

    d = Diagram("15-reference-map", "阅读索引：按问题选择资料", "入口均来自本节文末的参考表；关系连线表示阅读用途。", 564)
    rows = [(118, "先跑通训练服务", "PyTRIO 文档 / Tinker 对照 / SwanLab", CYAN), (215, "理解参数与相对反馈", "LoRA / GRPO（DeepSeekMath）\nDeepSeek-R1", BLUE), (312, "理解更新目标的变化", "PPO / DAPO / CISPO / GSPO", PURPLE), (409, "理解学生自己的轨迹", "GKD / MiniLLM / OPSD", AMBER)]
    for y, question, refs, color in rows:
        d.box(28, y, 276, 75, question, color=color)
        d.box(352, y, 460, 75, refs, color=color)
        d.arrow([(304, y+37), (352, y+37)])
    d.text(420, 530, "先用服务文档定位接口，再用原始论文核对目标函数、假设与实验条件。", "sub", "middle")
    d.save()


if __name__ == "__main__":
    build()
    print(f"Rebuilt 15 SVG diagrams in {OUT}")
