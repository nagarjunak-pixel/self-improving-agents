"""
worker.py — The Worker Agent
=============================
Executes individual subtasks produced by the Planner.

WHY THIS EXISTS:
The Worker is the "hands" of the system. While the Planner thinks about
structure, the Worker actually DOES the work — using tools, searching,
reasoning, and producing a concrete output.

KEY CONCEPT FOR STUDENTS:
In a multi-agent system, Workers are interchangeable and parallelizable.
You can spin up 5 Workers for 5 subtasks simultaneously. This is how
systems like Devin, SWE-Agent, and AutoGPT scale work.

This Worker inherits the full ReAct reasoning loop from BaseAgent,
so it can: think → use tools → observe → think → produce answer.
"""

from typing import Optional

from openai import OpenAI

from core.agent import BaseAgent
from core.tools import ToolRegistry
from core.memory import LongTermMemory

WORKER_PROMPT = """You are a WORKER agent in a multi-agent system.

Your job: Complete the specific subtask assigned to you. You have tools available — use them when needed.

Rules:
1. Be thorough. This is your one job — do it well.
2. Use tools when they help. Don't use tools when pure reasoning suffices.
3. Be concrete. "Research showed X is good" is bad. "Research showed X is good because (3 specific reasons with sources)" is good.
4. Your output will be reviewed by a CRITIC agent. Anticipate objections.
5. Keep your answer focused on your specific subtask — don't wander.

Produce a clear, well-structured answer."""


class WorkerAgent(BaseAgent):
    """A worker that executes subtasks using the ReAct loop."""

    def __init__(
        self,
        name: str = "Worker",
        model: str = "gpt-4o",
        tools: Optional[ToolRegistry] = None,
        long_term_memory: Optional[LongTermMemory] = None,
        client: Optional[OpenAI] = None,
        max_iterations: int = 8,
    ):
        super().__init__(
            name=name,
            system_prompt=WORKER_PROMPT,
            model=model,
            tools=tools,
            long_term_memory=long_term_memory,
            max_iterations=max_iterations,
            client=client,
        )