"""The Part 1 coding agent: fix a software issue and submit a git patch."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from assignment.agent.base import (
    DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
    DEFAULT_COMPACTION_MAX_TOKENS,
    Agent,
    format_tool_output,
)
from assignment.agent.tools import EXECUTE_TOOL, SEND_MESSAGE_TOOL
from assignment.env import Environment

# name -> handler(arguments) -> observation content
ToolHandler = Callable[[dict[str, Any]], str]


class CodeAgent(Agent):
    """An agent that fixes a software issue and submits a git patch."""

    def __init__(
        self,
        task: str,
        environment: Environment,
        model: str | None = None,
        logs_save_path: str | None = None,
        step_limit: int = 100,
        skills_path: str | None = None,
        auto_stop_environment: bool = True,
        compact_threshold_tokens: int | None = None,
        compaction_keep_recent_steps: int = DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
        compaction_max_tokens: int = DEFAULT_COMPACTION_MAX_TOKENS,
    ):
        super().__init__(
            environment=environment,
            model=model,
            logs_save_path=logs_save_path,
            step_limit=step_limit,
            skills_path=skills_path,
            auto_stop_environment=auto_stop_environment,
            compact_threshold_tokens=compact_threshold_tokens,
            compaction_keep_recent_steps=compaction_keep_recent_steps,
            compaction_max_tokens=compaction_max_tokens,
        )
        self.task = task
        self.submitted_patch = ""

        # TODO(Part 1.3): Make the `execute` and `send_message` tools available
        # to the agent.
        # Schema为声明给大模型的工具schema；handler 负责工具真正执行。
        # 新增工具时两者均需添加。
        self._tool_handlers: dict[str, ToolHandler] = {
            "execute": self._handle_execute,
            "send_message": self._handle_send_message,
        }
        self.tools.extend([EXECUTE_TOOL, SEND_MESSAGE_TOOL])
        
        if self.skills:
            # INVOKE_SKILL_TOOL is already registered by Agent.__init__.
            self._tool_handlers["invoke_skill"] = self._handle_invoke_skill

        # TODO(1.1.b): Construct the system prompt and task_prompt. These
        # should be usable by the `Agent.build_prompt` method.
        system_information = {
            "machine": self.env.machine,
            "release": self.env.release,
            "system": self.env.system,
            "version": self.env.version,
        }
        self.system_prompt = (
            "你是一个在 Linux 终端中工作的Code Agent。\n"
            "请使用可用工具检查代码库、编辑文件、运行命令并验证修改。"
            "优先通过具体的工具调用推进任务，而不是向用户提问。\n\n"
            "<system_information>\n"
            f"{json.dumps(system_information, indent=2)}\n"
            "</system_information>"
        )
        self.task_prompt = self.task

        # TODO(1.4): If any skills are available to the agent, make their
        # descriptions/metadata available to the agent in the prompt.
        if self.skills:
            catalog = "\n".join(skill["metadata"] for skill in self.skills.values())
            self.system_prompt += (
                "\n\n以下是可复用的Skills。\n"
                "当任务与某项技能的适用场景匹配时，请优先调用`invoke_skill`, 传入对应的技能名称，加载并阅读完整说明;"
                "然后遵循技能指引执行任务，而非采用默认处理流程。\n"
                f"\n<skills>\n{catalog}\n</skills>\n"
            )

    def _handle_execute(self, arguments: dict[str, Any]) -> str:
        """Run a sandbox command and format the terminal observation."""

        if "command" not in arguments:
            return "Error: execute requires a 'command' argument."
        result = self.env.execute(
            arguments["command"],
            timeout=arguments.get("timeout"),
            cwd=arguments.get("cwd"),
            env=arguments.get("env"),
            shell=arguments.get("shell"),
        )
        return format_tool_output(result)

    def _handle_send_message(self, arguments: dict[str, Any]) -> str:
        """Deliver a user-facing summary and mark the agent run finished."""

        summary = arguments.get("summary", "")
        self.finished = True
        return f"Message sent to the user:\n{summary}"

    def _handle_invoke_skill(self, arguments: dict[str, Any]) -> str:
        """Return one skill's full SKILL.md content (progressive disclosure)."""

        name = arguments.get("name")
        if not isinstance(name, str) or name not in self.skills:
            return f"Error: unknown skill {name!r}."
        return self.skills[name]["content"]

    def execute_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        """Execute ``execute`` and ``send_message`` calls in the code sandbox."""

        # TODO(Part 1.3): Parse each call, execute recognized tools, and return
        # one message per call (there may be multiple tool calls in one agent
        # response!). Malformed JSON and unknown tools must become recoverable
        # observations relayed to the agent instead of exceptions.
        observations: list[dict[str, str]] = []
        # tool_calls 是 OpenAI 格式的 dict 列表，例如：
        # {"id": "call_1", "type": "function",
        #  "function": {"name": "execute", "arguments": "{\"command\": \"ls\"}"}}
        # 不能写成 for call_id, tool_call in tool_calls（那是在解包 dict 的 key）。
        for call in tool_calls:
            call_id = str(call.get("id") or "")
            function = call.get("function") or {}
            name = function.get("name") or ""
            raw_arguments = function.get("arguments", "")

            try:
                # OpenAI tool-call arguments are a JSON string.
                arguments = json.loads(raw_arguments)
                if not isinstance(arguments, dict):
                    raise ValueError("tool arguments must be a JSON object")
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                observations.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": f"Error parsing tool arguments: {error}",
                    }
                )
                continue

            handler = self._tool_handlers.get(name)
            if handler is None:
                content = f"Error: unknown tool {name!r}."
            else:
                content = handler(arguments)

            observations.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": content,
                }
            )
        return observations
