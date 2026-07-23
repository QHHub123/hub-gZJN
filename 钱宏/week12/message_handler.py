from typing import Generator, List, Dict, Any
import json


class MessageHandler:
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt
        self.messages = []

    def initialize_messages(self, question: str) -> None:
        """初始化对话上下文"""
        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]

    def initialize_from_history(self, history_messages: List[Dict], question: str) -> None:
        """从历史记录初始化对话上下文，支持多轮对话"""
        self.messages = [{"role": "system", "content": self.system_prompt}]

        for msg in history_messages:
            msg_type = msg.get("type")
            if msg_type == "user":
                self.messages.append({
                    "role": "user",
                    "content": msg.get("content", ""),
                })

            elif msg_type == "action":
                assistant_msg = {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": f"call_{msg.get('step', 0)}",
                        "type": "function",
                        "function": {
                            "name": msg.get("action", ""),
                            "arguments": json.dumps(msg.get("action_input", {}), ensure_ascii=False),
                        }
                    }]
                }
                self.messages.append(assistant_msg)

                tool_msg = {
                    "role": "tool",
                    "tool_call_id": f"call_{msg.get('step', 0)}",
                    "content": str(msg.get("observation", "")),
                }
                self.messages.append(tool_msg)

            elif msg_type == "final":
                assistant_msg = {
                    "role": "assistant",
                    "content": msg.get("answer", ""),
                }
                self.messages.append(assistant_msg)

        self.messages.append({"role": "user", "content": question})

    def add_assistant_message(self, message: Any) -> None:
        """添加助手消息"""
        self.messages.append(message)

    def add_tool_result(self, tool_call_id: str, observation: str) -> None:
        """添加工具执行结果"""
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": str(observation),
        })

    def get_current_messages(self) -> List[Dict[str, Any]]:
        """获取当前消息列表"""
        return self.messages

    def add_manual_message(self, message: str, observation: Any) -> None:
        self.messages.append({"role": "assistant", "content": message})
        self.messages.append({
            "role": "user",
            "content": f"Observation: {observation}\n",
        })
