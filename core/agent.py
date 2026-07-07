"""
agent.py — Base Agent with ReAct Reasoning Loop
================================================
The foundation of the entire system.

WHY THIS EXISTS:
This is the CORE LOOP that every agent runs. It implements ReAct
(Reasoning + Acting), the most important pattern in agentic AI:

    ┌─────────────────────────────────┐
    │   THINK: "What should I do?"   │ ← LLM reasons about the situation
    │   ACT: Call a tool              │ ← LLM picks an action
    │   OBSERVE: See the result       │ ← Environment responds
    │   THINK: "What now?"            │ ← Loop back
    │   ...until done                 │
    └─────────────────────────────────┘

This is literally how ChatGPT, Claude, and every production agent works
under the hood. The loop is simple. The intelligence comes from:
  1. Good system prompts (what should the agent think about?)
  2. Good tools (what can the agent do?)
  3. Good memory (what does the agent remember?)
  4. Good stopping criteria (when to stop looping)

Read this file carefully. If you understand this loop, you understand
90% of agentic AI.
"""

import json
from typing import Optional

from openai import OpenAI

from core.memory import ShortTermMemory, LongTermMemory
from core.tools import ToolRegistry
from core.trace import TraceLogger, TraceStepType


class BaseAgent:
    """
    The base agent. All specialized agents (Planner, Worker, Critic, Judge)
    inherit from this.

    Core responsibilities:
      1. Maintain a conversation (ShortTermMemory)
      2. Call the LLM with tools available
      3. If the LLM wants to call a tool → call it → feed result back
      4. If the LLM gives a final answer → return it
      5. Log every step to the TraceLogger

    Subclasses customize:
      - system_prompt (their role/personality)
      - tools available to them
      - any special pre/post processing
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        model: str = "gpt-4o",
        tools: Optional[ToolRegistry] = None,
        long_term_memory: Optional[LongTermMemory] = None,
        max_iterations: int = 10,
        client: Optional[OpenAI] = None,
    ):
        self.name = name
        self.model = model
        self.tools = tools or ToolRegistry()
        self.long_term_memory = long_term_memory
        self.max_iterations = max_iterations
        self.client = client or OpenAI()
        self.trace = TraceLogger(agent_name=name)

        # Inject lessons from long-term memory into system prompt
        if long_term_memory and long_term_memory.lessons:
            lessons = long_term_memory.get_lessons()
            lessons_text = "\n".join(f"  - {l}" for l in lessons)
            system_prompt += f"\n\n## Lessons from past experience:\n{lessons_text}"

        self.memory = ShortTermMemory(system_prompt=system_prompt)

    def run(self, task: str) -> str:
        """
        THE REASONING LOOP. This is the heart of the agent.

        Args:
            task: What the agent should do

        Returns:
            The agent's final answer (string)
        """
        self.trace.log(TraceStepType.THINK, f"Starting task: {task}")
        self.memory.add("user", task)

        for i in range(self.max_iterations):
            self.trace.log(TraceStepType.THINK, f"Reasoning iteration {i+1}/{self.max_iterations}")

            # --- CALL THE LLM ---
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.memory.to_openai(),
                    tools=self.tools.schemas() if len(self.tools) > 0 else None,
                    temperature=0.7,
                )
                msg = response.choices[0].message
            except Exception as e:
                self.trace.log(TraceStepType.ERROR, f"LLM call failed: {e}")
                return f"Agent failed: {e}"

            # --- CHECK: DID THE LLM CALL A TOOL? ---
            if msg.tool_calls:
                # Add assistant message with tool calls to memory
                self.memory.add("assistant", msg.content or "")

                for tool_call in msg.tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        args = {}

                    self.trace.log(
                        TraceStepType.ACT,
                        f"Calling tool: {tool_name}",
                        data=args,
                    )

                    # --- EXECUTE THE TOOL ---
                    result = self.tools.execute(tool_name, **args)

                    self.trace.log(
                        TraceStepType.OBSERVE,
                        f"Tool result from {tool_name}",
                        data={"result_preview": result[:200]},
                    )

                    # --- FEED RESULT BACK TO LLM ---
                    self.memory.add("tool", result, tool_call_id=tool_call.id, name=tool_name)

                # Loop back → LLM will think again with the new information
                continue

            # --- NO TOOL CALL → FINAL ANSWER ---
            answer = msg.content or ""
            self.trace.log(TraceStepType.SUCCESS, f"Final answer produced ({len(answer)} chars)")
            self.memory.add("assistant", answer)
            return answer

        # Ran out of iterations
        self.trace.log(TraceStepType.ERROR, f"Hit max iterations ({self.max_iterations}) without final answer")
        return f"Agent could not complete the task in {self.max_iterations} iterations."

    def get_trace_summary(self) -> str:
        return self.trace.summary()