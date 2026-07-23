"""
Function Calling API 版 ReAct Agent

教学重点：
  1. 与手写版对比：框架帮你处理格式解析，但 Thought 过程在内部不可见
  2. tool_choice="auto" 让模型自己决定调用哪个工具或直接回答
  3. finish_reason 判断：tool_calls 表示继续调用，stop 表示给出最终答案
  4. 相同工具集，相同问题，对比两种实现的稳定性和步骤数

使用方式：
  python react_function_calling.py
  python react_function_calling.py --question "茅台近一年股价涨跌幅如何？"
  python react_function_calling.py --question "..." --max_steps 8

依赖：
  pip install openai faiss-cpu sentence-transformers akshare
  export DASHSCOPE_API_KEY="sk-xxx"
"""

import os
import json
import time
import logging
import argparse
from typing import Generator, List, Dict  # 用于声明 run() 是一个生成器函数，逐步 yield 结果

from openai import OpenAI  # OpenAI 兼容客户端，可对接 DeepSeek / DashScope 等兼容接口

from message_handler import MessageHandler

# 解决 FAISS / numpy 在某些环境下重复加载动态库导致的崩溃问题
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# 只保留 WARNING 级别日志，避免 INFO 级别刷屏干扰 ReAct 步骤输出
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ── 模型客户端配置（可切换为阿里云 DashScope） ──────────────────────────────────
# 切换方式：取消注释 DashScope 配置，注释掉 DeepSeek 配置即可
# client = OpenAI(
#     api_key=os.getenv("DASHSCOPE_API_KEY"),
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
# )
# MODEL = os.getenv("AGENT_MODEL", "qwen-max")
print("DEEPSEEK_API_KEY:", os.getenv("DEEPSEEK_API_KEY"))

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),   # 从环境变量读取 API Key
    base_url="https://api.deepseek.com",      # DeepSeek 的 OpenAI 兼容端点
)
MODEL = os.getenv("AGENT_MODEL", "deepseek-v4-flash")  # 默认使用轻量版，可通过环境变量覆盖

# ── 系统提示词：约束模型的工具调用顺序和回答规范 ─────────────────────────────────
FC_SYSTEM_PROMPT = """你是一个专业的A股金融分析助手。
规则：
- 调用 financial_indicator 或 stock_price 之前，必须先用 company_lookup 获取股票代码
- 数字计算必须使用 calculator 工具，不能心算
- Final Answer 必须引用具体数据来源
- 如果没有合适工具能回答，直接说明原因
"""


def run(question: str, max_steps: int = 10, history_messages: List[Dict] = None) -> Generator[dict, None, None]:
    """
    执行 Function Calling 版 ReAct 循环，yield 每一步结构化结果

    格式与 react_manual.run() 保持一致，便于 evaluate.py 统一对比

    核心流程：
      1. 将问题放入 messages，调用模型
      2. 若模型返回 tool_calls → 执行对应工具，将结果追加到 messages，继续循环
      3. 若模型直接返回文本（finish_reason=stop）→ 输出最终答案，结束循环
      4. 若达到 max_steps 仍未结束 → 返回超步警告

    参数：
      question: 用户问题
      max_steps: 最大步数
      history_messages: 历史对话消息列表，用于多轮对话
    """
    from tools import TOOLS_MAP, TOOLS_SCHEMA

    message_handler = MessageHandler(FC_SYSTEM_PROMPT)

    if history_messages:
        message_handler.initialize_from_history(history_messages, question)
    else:
        message_handler.initialize_messages(question)

    # ── ReAct 主循环：每一步都可能是一次工具调用或最终回答 ──────────────────────
    for step in range(1, max_steps + 1):
        # 获取当前消息上下文
        messages = message_handler.get_current_messages()

        # 向模型发送当前对话上下文，附带工具描述，让模型自主决定是否调用工具
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS_SCHEMA,       # 告诉模型有哪些工具可用（JSON Schema 格式）
            tool_choice="auto",       # "auto" = 模型自行判断：需要工具就调用，否则直接回答
            temperature=0,            # 温度 0 保证输出稳定、可复现
        )
        msg    = response.choices[0].message       # 模型返回的消息对象
        reason = response.choices[0].finish_reason  # 结束原因："stop"=直接回答, "tool_calls"=请求调用工具

        # ── 分支 1：模型决定直接回答（无工具调用） ────────────────────────────────
        if reason == "stop" or not msg.tool_calls:
            yield {
                "step":   step,
                "type":   "final",
                "thought": "",   # FC 版的推理过程在模型内部，无法提取
                "answer": msg.content or "（模型返回空内容）",
            }
            return  # 得到最终答案，结束生成器

        # ── 分支 2：模型请求调用工具 → 将模型的完整消息追加到上下文 ────────────────
        # 必须追加 msg 本身（包含 tool_calls 信息），否则下一轮模型不知道之前调用了什么
        message_handler.add_assistant_message(msg)

        # 依次处理模型请求的每一个工具调用（模型可能一次请求多个工具）
        for tool_call in msg.tool_calls:
            tool_name = tool_call.function.name          # 模型选择的工具名称
            try:
                tool_args = json.loads(tool_call.function.arguments)  # 模型生成的工具参数（JSON 字符串→字典）
            except json.JSONDecodeError:
                tool_args = {}  # 模型偶尔会生成非法 JSON，降级为空参数

            # 从工具映射表中查找对应的函数
            tool_fn = TOOLS_MAP.get(tool_name)
            if tool_fn is None:
                observation = f"未知工具 '{tool_name}'"  # 模型幻觉：调用了不存在的工具
            else:
                try:
                    observation = tool_fn(**tool_args)    # 执行工具函数，传入模型生成的参数
                except TypeError as e:
                    observation = f"工具参数错误: {e}"    # 参数不匹配时的错误提示

            # 将本次工具调用的完整结果结构化，yield 给调用方（便于 evaluate.py 逐步追踪）
            step_result = {
                "step":         step,
                "type":         "action",
                "thought":      "",   # Function Calling 版 Thought 在模型内部，不可见
                "action":       tool_name,
                "action_input": tool_args,
                "observation":  str(observation),
            }
            yield step_result

            # 将工具执行结果追加到对话上下文，role="tool" 是 OpenAI 规定的工具回复格式
            # tool_call_id 用于关联本次回复与之前模型发出的 tool_call 请求
            # 添加工具结果到消息上下文
            message_handler.add_tool_result(tool_call.id, observation)

    # ── 超步保护：循环结束仍未得到最终答案 ──────────────────────────────────────
    yield {
        "step":   max_steps + 1,
        "type":   "max_steps",
        "answer": f"已达最大步数 {max_steps}，未能得出最终答案",
    }


# ── CLI 彩色输出工具 ────────────────────────────────────────────────────────────
# ANSI 转义码映射：每种信息类型对应一种颜色，增强终端可读性

COLORS = {
    "thought": "\033[36m",  # 青色 — 推理过程
    "action":  "\033[33m",  # 黄色 — 工具调用
    "obs":     "\033[32m",  # 绿色 — 观察结果
    "final":   "\033[35m",  # 紫色 — 最终答案
    "error":   "\033[31m",  # 红色 — 错误/警告
    "reset":   "\033[0m",   # 重置颜色
}

def _c(color: str, text: str) -> str:
    """给文本包裹 ANSI 颜色码，终端中显示彩色输出"""
    return f"{COLORS[color]}{text}{COLORS['reset']}"


def run_and_print(question: str, max_steps: int = 10):
    """运行 ReAct 循环并在终端以彩色格式逐步打印每一步的结果"""
    print(f"\n{'='*60}")
    print(f"问题: {question}")
    print(f"模型: {MODEL}  实现: Function Calling")
    print('='*60)

    start = time.time()  # 记录开始时间，用于计算总耗时

    for step_data in run(question, max_steps=max_steps):
        stype = step_data["type"]

        if stype == "action":
            # 打印工具调用步骤：Thought → Action → Observation
            print(f"\n[Step {step_data['step']}]")
            # Thought 在 FC 版不可见，显示提示
            print(_c("thought", "🧠 Thought: （模型内部推理，Function Calling 版不可见）"))
            print(_c("action",  f"🔧 Action:  {step_data['action']}"))
            print(_c("action",  f"   Input:   {json.dumps(step_data['action_input'], ensure_ascii=False)}"))
            print(_c("obs",     f"👁  Obs:     {step_data['observation'][:300]}"))  # 截断过长输出，最多显示 300 字符

        elif stype == "final":
            # 打印最终答案和统计信息
            elapsed = time.time() - start
            print(f"\n{'─'*60}")
            print(_c("final", f"\n✅ Final Answer:\n{step_data['answer']}"))
            print(f"\n共 {step_data['step']} 步，耗时 {elapsed:.1f}s")

        elif stype in ("error", "max_steps"):
            # 打印错误或超步警告
            print(_c("error", f"\n⚠️  {step_data.get('answer', '')}"))


# ── 命令行入口 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question",  default="贵州茅台和五粮液2023年的毛利率哪家更高？差多少个百分点？")
    parser.add_argument("--max_steps", type=int, default=10)
    args = parser.parse_args()
    run_and_print(args.question, args.max_steps)
