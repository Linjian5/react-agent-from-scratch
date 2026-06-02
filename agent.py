"""
手写 ReAct Agent —— 不依赖任何 Agent 框架，纯 OpenAI 兼容 API 实现
Thought -> Action -> Observation 循环。

设计要点：
- stop=["Observation:"] 让 LLM 停在动作处，由我们执行工具后回填真实观察，防止 LLM 自己编造 Observation
- 最大步数 + 重复动作检测，双重防死循环
- finish[answer] 作为显式终止动作
"""
import os
import re
import sys

from dotenv import load_dotenv
from openai import OpenAI

from tools import TOOL_REGISTRY, ToolError

# 强制 stdout/stderr 用 UTF-8，避免 Windows GBK 终端无法显示 ✅ / °C 等字符
try:
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass  # 老版本 Python 或非 TTY 环境，无所谓

load_dotenv()

API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
MODEL = os.getenv("MODEL", "deepseek-chat")
MAX_STEPS = int(os.getenv("MAX_STEPS", "8"))

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ReAct few-shot prompt：教 LLM 严格按 Thought/Action/Action Input 格式输出
SYSTEM_PROMPT = """你是一个 ReAct 风格的 Agent。你必须严格按以下格式逐步推理：

Thought: 你对当前情况的思考
Action: 要使用的工具名（必须是可用工具之一）
Action Input: 工具的输入参数

系统会执行工具并返回：
Observation: 工具执行结果

然后你继续下一轮 Thought / Action / Action Input，直到你能回答用户问题。
当你已经得到最终答案时，用以下格式结束：

Thought: 我已经知道答案了
Action: finish
Action Input: 给用户的最终回答

可用工具：
{tools}

规则：
1. 一次只输出一个 Thought + 一个 Action + 一个 Action Input，然后停下等待 Observation。
2. 不要自己编造 Observation。
3. Action 必须是可用工具名或 finish。
4. 信息足够时立刻 finish，不要重复调用同一工具。
"""


def build_tools_desc() -> str:
    lines = []
    for name, meta in TOOL_REGISTRY.items():
        lines.append(f"- {name}: {meta['description']}")
    return "\n".join(lines)


def parse_action(text: str):
    """从 LLM 输出里抽取 Action 和 Action Input。"""
    action_match = re.search(r"Action:\s*(.+?)\s*(?:\n|$)", text)
    input_match = re.search(r"Action Input:\s*(.+?)\s*(?:\n|$)", text, re.DOTALL)
    if not action_match:
        return None, None
    action = action_match.group(1).strip()
    action_input = input_match.group(1).strip() if input_match else ""
    return action, action_input


def run(question: str) -> str:
    system = SYSTEM_PROMPT.format(tools=build_tools_desc())
    # scratchpad 累积 ReAct 轨迹
    scratchpad = f"Question: {question}\n"
    seen_actions = []  # 重复动作检测

    for step in range(1, MAX_STEPS + 1):
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": scratchpad},
            ],
            stop=["Observation:"],  # ⭐ 关键：让 LLM 停在动作处，不要自己编 Observation
            temperature=0,
        )
        output = resp.choices[0].message.content.strip()
        print(f"\n--- Step {step} ---\n{output}")
        scratchpad += output + "\n"

        action, action_input = parse_action(output)

        if action is None:
            # LLM 没按格式输出，提示纠正
            scratchpad += "Observation: 格式错误，请按 Action: <工具> / Action Input: <输入> 格式输出。\n"
            continue

        if action == "finish":
            return action_input

        # 重复动作检测：连续相同 action+input 直接终止
        sig = f"{action}::{action_input}"
        if seen_actions[-2:] == [sig, sig]:
            return f"（检测到重复动作，提前终止）最近一次结果见上。"
        seen_actions.append(sig)

        # 执行工具
        tool = TOOL_REGISTRY.get(action)
        if tool is None:
            obs = f"ToolError: 未知工具 '{action}'，可用：{', '.join(TOOL_REGISTRY)}"
        else:
            try:
                obs = str(tool["fn"](action_input))
            except ToolError as e:
                obs = f"ToolError: {e}"
            except Exception as e:
                obs = f"UnexpectedError: {type(e).__name__}: {e}"

        print(f"Observation: {obs}")
        scratchpad += f"Observation: {obs}\n"

    return f"（达到最大步数 {MAX_STEPS}，未能得出最终答案）"


def main():
    if not API_KEY:
        print("ERROR: 未设置 DEEPSEEK_API_KEY / OPENAI_API_KEY，请先 copy .env.example 到 .env")
        sys.exit(1)
    print(f"手写 ReAct Agent — model={MODEL} @ {BASE_URL}")
    print(f"工具：{', '.join(TOOL_REGISTRY)}")
    print("输入问题，'exit' 退出。\n")
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit", ":q"):
            break
        answer = run(q)
        print(f"\n✅ Final Answer: {answer}\n")


if __name__ == "__main__":
    main()
