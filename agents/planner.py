"""
planner.py — The Planner Agent
===============================
Breaks a complex task into smaller subtasks.

WHY THIS EXISTS:
Real tasks are too big for one agent to handle in a single reasoning loop.
"Design a web app" → too vague. But:
  1. "Identify the core features needed"
  2. "Choose the tech stack"
  3. "Design the database schema"
  → now each is tractable.

The Planner is the "project manager" of the system. It doesn't DO the work —
it decides WHAT work needs to be done and in what order.

KEY CONCEPT FOR STUDENTS:
This is how production multi-agent systems work. The "orchestrator" or
"planner" agent decomposes the problem, and "worker" agents execute.
Same pattern used by AutoGPT, Devin, etc.
"""

import json
from typing import Optional

from openai import OpenAI

from core.agent import BaseAgent
from core.trace import TraceLogger, TraceStepType

PLANNER_PROMPT = """You are a PLANNER agent in a multi-agent system.

Your job: Take a complex task and break it into 2-5 subtasks that other agents can work on independently.

Rules:
1. Each subtask must be self-contained (an agent can do it without seeing the others)
2. Order matters — list them in the order they should be done
3. Be specific. "Research" is bad. "Research the best Python frameworks for real-time websockets and list 3 with pros/cons" is good.
4. If the task is simple enough for one agent, return just one subtask (the original task).

Output format (STRICT JSON):
```json
{
  "subtasks": [
    {"id": 1, "task": "...", "description": "..."},
    ...
  ]
}
```

Output ONLY the JSON. No preamble, no explanation."""


class PlannerAgent(BaseAgent):
    """Plans a task by decomposing it into subtasks."""

    def __init__(self, model: str = "gpt-4o", client: Optional[OpenAI] = None, **kwargs):
        super().__init__(
            name="Planner",
            system_prompt=PLANNER_PROMPT,
            model=model,
            tools=None,  # Planner doesn't use tools — it just thinks
            max_iterations=1,  # One shot — it should respond immediately
            client=client,
            **kwargs,
        )

    def plan(self, task: str) -> list[dict]:
        """
        Decompose a task into subtasks.

        Returns:
            List of {"id": int, "task": str, "description": str}
        """
        self.trace.log(TraceStepType.PLAN, f"Decomposing: {task}")
        raw = self.run(task)

        # Parse the JSON from the response
        try:
            # Extract JSON from potential markdown code block
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0]
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0]

            data = json.loads(raw.strip())
            subtasks = data.get("subtasks", [])
            self.trace.log(
                TraceStepType.SUCCESS,
                f"Decomposed into {len(subtasks)} subtasks",
                data=subtasks,
            )
            return subtasks
        except (json.JSONDecodeError, IndexError) as e:
            self.trace.log(
                TraceStepType.ERROR,
                f"Failed to parse plan JSON: {e}",
                data={"raw": raw[:300]},
            )
            # Fallback: treat the whole thing as one task
            return [{"id": 1, "task": task, "description": task}]