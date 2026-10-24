"""The domain-independent ReAct loop shared by both agents.

Part 1 completes the generic loop here; the two subclasses in this package
supply only their own tools and tool executors.
"""

from __future__ import annotations

from copy import deepcopy
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from openai import OpenAI

from assignment.env import Environment
from assignment.agent.tools import INVOKE_SKILL_TOOL

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_COMPACTION_KEEP_RECENT_STEPS = 1
DEFAULT_COMPACTION_MAX_TOKENS = 1_200
MAX_OBSERVATION_CHARS = 10_000

# TODO(Part 2): Write instructions that make the model produce concise working
# memory for a software agent. The prompt should preserve concrete progress,
# failures, test results, constraints, and next steps without copying raw output.
COMPACTION_SYSTEM_PROMPT = """\
你是一位 Agent 上下文压缩专家。你的任务是将 Agent 的历史交互记录压缩为简洁、准确、可执行的工作记忆，使 Agent 能够基于压缩后的上下文无缝继续任务。

**必须保留的信息：**
- **目标与约束**：用户的任务目标、具体要求、偏好及明确限制。
- **当前状态**：已完成的工作、已确定的决策，以及任务的最新进展。
- **操作与结果**：已查看或修改的文件、执行过的命令与工具调用、关键输出及已验证的结论。
- **失败与阻塞**：失败的尝试、已知问题、未解决的疑问，以及避免重复犯错的重要经验。
- **后续行动**：尚未完成的任务，以及下一步最直接、可执行的操作。

**压缩规则：**
- 优先保留对后续任务执行有价值的信息，而非追求历史记录的完整性。
- 对后续执行至关重要的文件路径、标识符、命令、参数及错误信息，必须准确保留。
- 明确区分已验证的事实、推测、假设及尚未验证的结果。
- 当历史信息发生变化或冲突时，以最新的有效状态为准，删除已失效的信息。
- 删除重复讨论、中间推理、原始工具日志，以及冗长的代码和文件内容。
- 使用简洁、客观的要点，按主题组织信息。
- 不得编造信息，不得将未经验证的操作描述为已成功完成。
- 仅压缩已有历史，不执行实际任务，也不提出超出历史依据的新计划。

**输出要求：**

仅输出压缩后的工作记忆，不添加前言、解释或其他无关内容。
"""


class StepLimitError(Exception):
    """Raised when an agent exhausts its model-call budget."""


def format_tool_output(output: dict[str, Any]) -> str:
    """Format a terminal result as a compact, tagged model observation."""

    elements: list[str] = []
    for key, value in sorted(output.items()):
        if isinstance(value, str) and len(value) > MAX_OBSERVATION_CHARS:
            # Leave room for the elision notice so the formatted value itself,
            # not just its retained source slices, stays below the limit.
            # 保留前4900个字符和后4900个字符, 中间用省略号表示
            retained_at_each_end = 4_900
            omitted = len(value) - (2 * retained_at_each_end)
            value = (
                f"{value[:retained_at_each_end]}\n"
                f"[{omitted} characters elided; read a narrower range]\n"
                f"{value[-retained_at_each_end:]}"
            )
        elements.append(f"<{key}>{value}</{key}>")
    return "\n".join(elements)


def rough_message_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate prompt tokens without a provider-specific tokenizer."""

    # 估算整段 messages 的总 token 数：序列化后按约 4 个字符 = 1 token快速计算
    serialized = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
    return max(1, math.ceil(len(serialized) / 4))


class Agent:
    """Base class for a ReAct agent with pluggable tools."""

    def __init__(
        self,
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
        self.env = environment
        self.model = model or os.environ.get("OPENAI_MODEL")
        if not self.model:
            raise RuntimeError("OPENAI_MODEL is not set.")

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        base_url = os.environ.get("OPENAI_BASE_URL")
        if not base_url:
            raise RuntimeError("OPENAI_BASE_URL is not set.")
        try:
            max_retries = int(os.environ.get("OPENAI_MAX_RETRIES", "5"))
        except ValueError as exc:
            raise RuntimeError("OPENAI_MAX_RETRIES must be an integer.") from exc
        if max_retries < 0:
            raise RuntimeError("OPENAI_MAX_RETRIES must be non-negative.")

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=max_retries,
        )

        self.logs_save_path = logs_save_path
        self.step_limit = step_limit
        self.auto_stop_environment = auto_stop_environment
        if compact_threshold_tokens is not None and compact_threshold_tokens <= 0:
            raise ValueError("compact_threshold_tokens must be positive or None")
        if (
            compaction_keep_recent_steps is not None
            and compaction_keep_recent_steps < 1
        ):
            raise ValueError("compaction_keep_recent_steps must be at least 1")
        if compaction_max_tokens is not None and compaction_max_tokens < 1:
            raise ValueError("compaction_max_tokens must be positive")
        # A None threshold turns compaction off. The other two settings then
        # describe a compaction that never happens, so fall back to the
        # defaults rather than leaving a None for later code to trip over.
        self.compact_threshold_tokens = compact_threshold_tokens
        self.compaction_keep_recent_steps = (
            DEFAULT_COMPACTION_KEEP_RECENT_STEPS
            if compaction_keep_recent_steps is None
            else compaction_keep_recent_steps
        )
        self.compaction_max_tokens = (
            DEFAULT_COMPACTION_MAX_TOKENS
            if compaction_max_tokens is None
            else compaction_max_tokens
        )

        # Each agent supplies its own opening messages: the standing
        # instructions, and the task statement that starts the run.
        # 此处system_prompt与task_prompt使用空字符串占位,
        # 各子类必须自己填充system_prompt 和 task_prompt
        self.system_prompt: str = ""
        self.task_prompt: str = ""

        self.api_prompts: list[list[dict[str, Any]]] = []
        self.api_responses: list[dict[str, Any]] = []
        self.compaction_events: list[dict[str, Any]] = []
        self.tools: list[dict[str, Any]] = []
        self.finished = False
        self.steps_taken = 0

        self.skills_path = Path(skills_path) if skills_path is not None else None
        self.skills: dict[str, dict[str, str]] = (
            self.load_skills(self.skills_path) if self.skills_path is not None else {}
        )

        if self.skills:
            self.tools.append(INVOKE_SKILL_TOOL)

        # TODO(1.1.a): Add machinery to maintain agent state as it takes actions
        # and observes the results.

        # 注意:
        # 此处self.messages 只保存assistant actions 与 tool observations, 
        # 不包含system_prompt / task_prompt, 
        # system_prompt / task_prompt 由子类单独设置，在 build_prompt 时再拼到前面.
        self.messages: list[dict[str, Any]] = []

    def load_skills(self, skills_path: Path) -> dict[str, dict[str, str]]:
        """Load the skill folders exposed to this agent."""

        # TODO(1.4): Validate ``skills_path``, discover one ``SKILL.md``
        # per child directory, parse its YAML frontmatter (what's between the
        # `---` tags at the head of the file), and return a mapping
        # keyed by the frontmatter ``name``. Each value must contain a concise
        # ``metadata`` string for the model's skill catalog and the complete
        # ``content`` of the skill file for ``invoke_skill``. Reject duplicate
        # names and malformed or missing frontmatter with a clear
        # ``ValueError``.
        if not skills_path.exists() or not skills_path.is_dir():
            raise ValueError(f"skills_path does not exist or is not a directory: {skills_path}")

        skills: dict[str, dict[str, str]] = {}
        for child_dir in sorted(skills_path.iterdir()):
            if not child_dir.is_dir():
                continue
            skill_file = child_dir / "SKILL.md"
            if not skill_file.is_file():
                raise ValueError(
                    f"skill directory {child_dir.name} is missing SKILL.md"
                )

            content = skill_file.read_text(encoding="utf-8")
            if not content.startswith("---"):
                raise ValueError(
                    f"SKILL.md in {child_dir.name} is missing YAML frontmatter"
                )
            parts = content.split("---", 2)
            # Expect: "" | frontmatter | body... after split on first two fences.
            if len(parts) < 3:
                raise ValueError(
                    f"SKILL.md in {child_dir.name} has malformed YAML frontmatter"
                )
            try:
                frontmatter = yaml.safe_load(parts[1])
            except yaml.YAMLError as exc:
                raise ValueError(
                    f"SKILL.md in {child_dir.name} has invalid YAML frontmatter: {exc}"
                ) from exc
            if not isinstance(frontmatter, dict):
                raise ValueError(
                    f"SKILL.md in {child_dir.name} frontmatter must be a mapping"
                )
            name = frontmatter.get("name")
            description = frontmatter.get("description")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    f"SKILL.md in {child_dir.name} is missing a valid frontmatter name"
                )
            if not isinstance(description, str) or not description.strip():
                raise ValueError(
                    f"SKILL.md in {child_dir.name} is missing a valid frontmatter "
                    "description"
                )
            if name in skills:
                raise ValueError(f"duplicate skill name {name!r}")

            skills[name] = {
                "metadata": f"name: {name}\ndescription: {description}",
                "content": content,
            }
        return skills

    def query_language_model(self) -> dict[str, Any]:
        """Send one tool-enabled Chat Completions request and normalize it."""

        messages = self.build_prompt()
        self.api_prompts.append(deepcopy(messages))
        step_number = self.steps_taken + 1
        print(
            f"[agent] step {step_number}/{self.step_limit}: requesting action",
            flush=True,
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                reasoning_effort="medium",
                max_completion_tokens=4096,
            )
        except Exception as exc:
            print(
                f"[agent] step {step_number}: model request failed after retries "
                f"({type(exc).__name__}: {exc})",
                flush=True,
            )
            raise
        self.api_responses.append(response.model_dump(mode="json"))
        self.steps_taken += 1
        message = self.process_response(response)
        tool_names = [
            call.get("function", {}).get("name", "unknown")
            for call in message.get("tool_calls", [])
            if isinstance(call, dict)
        ]
        if tool_names:
            print(
                f"[agent] step {step_number}: tool call(s): {', '.join(tool_names)}",
                flush=True,
            )
        else:
            print(
                f"[agent] step {step_number}: response contained no parsed tool call; "
                "the loop should preserve the response and continue",
                flush=True,
            )
        return message

    def process_response(self, response: Any) -> dict[str, Any]:
        """Return relevant parts of the language model's response."""

        return response.choices[0].message.model_dump(exclude_none=True)

    def build_prompt(self) -> list[dict[str, Any]]:
        # TODO(1.1.a): Construct a sequence of messages that form the language
        # model prompt. This should include standing instructions, task
        # specification, prior interaction including observations, reasoning,
        # and actions from previous turns. Note that this method should be
        # domain-agnostic and construct the prompt in a way that would apply
        # to any of the inheriting domain-specific agents.

        # You want to be careful about which attributes of the class you modify
        # here as they may also be handled by the subclasses.
        # 拼出发给模型的完整 messages：
        # - system：子类提供的站立指令（system_prompt）
        # - user：整场任务说明（task_prompt），一般不变
        # - *self.messages：此前各轮的 assistant 动作与 tool 观察（由 run 循环追加）
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.task_prompt},
            *self.messages,
        ]

    def estimate_active_prompt_tokens(self) -> int:
        """Estimate the next prompt, calibrated by the provider's latest usage."""

        current_prompt = self.build_prompt()
        rough_current = rough_message_tokens(current_prompt)

        # 如果还没有任何API调用记录, 则返回current_prompt的token数, 即rough_current
        if not self.api_prompts or not self.api_responses:
            return rough_current

        # 如果已经有API调用记录, 则返回当前rough_current与上次API调用后的token数之和
        usage = self.api_responses[-1].get("usage") or {}
        actual_previous = usage.get("prompt_tokens")
        if not isinstance(actual_previous, int):
            return rough_current

        rough_previous = rough_message_tokens(self.api_prompts[-1])
        added_since_previous_request = max(0, rough_current - rough_previous)
        return actual_previous + added_since_previous_request

    @property
    def compaction_enabled(self) -> bool:
        """Whether this agent compacts its context at all."""

        return self.compact_threshold_tokens is not None

    def _split_messages_for_compaction(
        self,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Split history into an old prefix to summarize and recent turns to keep."""

        # self.messages 是按「轮」组织的，而不是按条数：
        #   assistant(+tool_calls) → 若干 tool 观察 → assistant → tool* → ...
        # 压缩也按轮切：保留最近 keep 个完整 assistant 回合（含其后续 tool），
        # 更早的轮次整段拿去摘要。因此用 assistant 下标当锚点，
        # 避免按「最后 N 条消息」切开把某次 tool 和它的 assistant 拆散。
        assistant_indices = [
            index
            for index, message in enumerate(self.messages)
            if message.get("role") == "assistant"
        ]
        keep = self.compaction_keep_recent_steps
        # 最近 keep 轮的起点 = 倒数第 keep 个 assistant 的位置。
        first_recent = assistant_indices[-keep]
        return self.messages[:first_recent], self.messages[first_recent:]

    def compact_context(self):
        """Replace parts of prompt with model-generated working memory. Changes the
        content that `build_prompt` emits."""

        # TODO(2.1): Prompt the model to compact the context. The system
        # prompt should ask for concise factual working memory and preserve
        # the objective, constraints, files, commands, edits, concrete
        # results, failed approaches, tests, blockers, and next action.
        # Summarize only an old prefix; retain the original system/task
        # messages verbatim and at least the latest complete assistant action
        # with all linked tool observations. The resulting summary should change
        # what `build_prompt` emits, and reduce the length of the prompt.

        old_messages, recent_messages = self._split_messages_for_compaction()
        compaction_task_message = {
            "role": "user",
            "content": (
                "请将上述历史交互压缩为简洁、准确的工作记忆，保留继续完成当前任务所必需的关键信息，"
                "确保 Agent 能够基于压缩后的上下文无缝恢复执行。"
            ),
        }

        compaction_prompt = [
            {"role": "system", "content": COMPACTION_SYSTEM_PROMPT},
            {"role": "user", "content": self.task_prompt},
            *old_messages,
            compaction_task_message,
        ]

        ### Do not modify this section ###
        compaction_response = self.client.chat.completions.create(
            model=self.model,
            messages=compaction_prompt,
            reasoning_effort="medium",
            max_completion_tokens=self.compaction_max_tokens,
        )
        ##################################

        # Use `compaction_response` to update what `build_prompt` emits, but
        # DO NOT modify the object itself. Let the method return it unchanged.
        summary = compaction_response.choices[0].message.content or ""
        # system_prompt / task_prompt stay as-is in build_prompt; only replace
        # the old transcript prefix with working memory, then keep recent turns.
        self.messages = [
            {
                "role": "user",
                "content": f"<working_memory>\n{summary}\n</working_memory>",
            },
            *recent_messages,
        ]

        ### Do not modify this section ###
        return compaction_prompt, compaction_response.model_dump(mode="json")
        ##################################

    def maybe_compact_context(self) -> bool:
        """Compact before the next action request when the threshold is reached."""

        if not self.compaction_enabled:
            return False

        # Context too short to compact yet
        if self.estimate_active_prompt_tokens() < self.compact_threshold_tokens:
            return False

        prompt_before = deepcopy(self.build_prompt())

        # Not enough steps (each assistant turn corresponds to a step) to force
        # compaction yet
        if (
            len([m for m in prompt_before if m.get("role") == "assistant"])
            <= self.compaction_keep_recent_steps
        ):
            return False

        compaction_prompt, compaction_response = self.compact_context()
        prompt_after = deepcopy(self.build_prompt())
        self.compaction_events.append(
            {
                "step": self.steps_taken,
                "estimated_tokens_before": rough_message_tokens(prompt_before),
                "estimated_tokens_after": rough_message_tokens(prompt_after),
                "active_prompt_before": deepcopy(prompt_before),
                "compaction_prompt": compaction_prompt,
                "compaction_response": compaction_response,
            }
        )
        return True

    def run(self) -> None:
        """Run ReAct steps, always saving the trajectory and stopping Modal."""

        try:
            # TODO(1.2) Run the ReAct loop. Orchestrate the sequence of
            # prompting the language model to produce reasoning and actions,
            # extracting the tool calls produced by the model, and executing
            # the tool calls to obtain the agent's observation for the next
            # step. Ensure you identify when the agent has completed the task
            # by setting `Agent.finished`. If the agent exceeds the
            # `step_limit`, raise `StepLimitError`.
            while not self.finished:
                if self.steps_taken >= self.step_limit:
                    raise StepLimitError(
                        f"Agent exceeded step_limit={self.step_limit}."
                    )

                # TODO(2.2) Call `maybe_compact_context()` before each new action
                # request in your shared loop. It already estimates active tokens
                # and handles the threshold, and tracks compaction events for
                # logging.
                # 仅返回是否发生了压缩（True/False）；压缩结果写进内部状态，
                self.maybe_compact_context()

                assistant_message = self.query_language_model()
                self.messages.append(assistant_message)

                tool_calls = assistant_message.get("tool_calls") or []
                if tool_calls:
                    # Subclasses set `finished` inside execute_tool_calls when
                    # the domain signals completion (e.g. send_message / game_over).
                    observations = self.execute_tool_calls(tool_calls)
                    self.messages.extend(observations)
        finally:
            # This block is provided infrastructure. Do not modify it: a
            # trajectory is required even when a run fails.
            if self.logs_save_path:
                path = Path(self.logs_save_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(
                        {
                            "prompts": self.api_prompts,
                            "responses": self.api_responses,
                            "compactions": self.compaction_events,
                        },
                        indent=2,
                    )
                )
            if self.auto_stop_environment:
                stop = getattr(self.env, "stop", None)
                if callable(stop):
                    stop()

    def execute_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        """Execute domain-specific calls and return linked tool observations."""

        # You do not need to implement anything here. This method is
        # domain-specific and implemented by the relevant subclasses
        raise NotImplementedError
