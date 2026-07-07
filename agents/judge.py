"""
judge.py — The Judge Agent (Final Synthesis)
=============================================
Takes all worker outputs and synthesizes a final, polished answer.

WHY THIS EXISTS:
When multiple agents produce multiple pieces of work, someone needs to
stitch them together into a coherent final output. That's the Judge.

The Judge is NOT just concatenating outputs. It:
  1. Reads all subtask results
  2. Identifies conflicts between them
  3. Resolves conflicts (picks the best version)
  4. Synthesizes into a single coherent answer
  5. Scores the overall quality

This maps to the "aggregator" or "synthesizer" pattern in production
multi-agent systems. It's also related to "Agent-as-a-Judge" — the
Judge is an agent that evaluates the overall system output.

KEY CONCEPT FOR STUDENTS:
In a real system, the Judge would also decide:
  - Do we need another round of work?
  - Which subtask needs more effort?
  - Is the overall answer good enough to ship?

Here, the Judge focuses on synthesis + scoring, keeping it simple
while still teaching the pattern.
"""

from typing import Optional

from openai import OpenAI

from core.agent import BaseAgent
from core.trace import TraceStepType

JUDGE_PROMPT = """You are the JUDGE agent in a multi-agent system.

Your job: Take the outputs of multiple worker agents and synthesize them into ONE final, coherent answer.

Rules:
1. Do NOT just concatenate. SYNTHESIZE — create a unified, flowing answer.
2. If workers contradict each other, use your judgment to pick the best version and note the conflict.
3. Structure the final answer clearly (headers, bullet points, etc.)
4. At the end, give a quality score from 1-10 with a brief justification.

Output format:
[Your synthesized answer here]

---
**Quality Score: X/10**
**Justification: [brief reasoning]**
"""

JUDGE_EVALUATION_PROMPT = """## Original Task:
{task}

## Worker Outputs (by subtask):
{worker_outputs}

Synthesize these into a final answer. Remember: unify, don't concatenate."""


class JudgeAgent(BaseAgent):
    """Synthesizes multiple worker outputs into a final answer."""

    def __init__(self, model: str = "gpt-4o", client: Optional[OpenAI] = None, **kwargs):
        super().__init__(
            name="Judge",
            system_prompt=JUDGE_PROMPT,
            model=model,
            tools=None,
            max_iterations=1,
            client=client,
            **kwargs,
        )

    def synthesize(self, task: str, worker_outputs: list[dict]) -> str:
        """
        Synthesize multiple worker outputs into a final answer.

        Args:
            task: The original task
            worker_outputs: List of {"subtask": str, "output": str}

        Returns:
            The final synthesized answer (with quality score)
        """
        self.trace.log(
            TraceStepType.DECIDE,
            f"Synthesizing {len(worker_outputs)} worker outputs",
        )

        # Format worker outputs for the judge
        outputs_text = ""
        for w in worker_outputs:
            outputs_text += f"\n### Subtask: {w['subtask']}\n{w['output']}\n"

        prompt = JUDGE_EVALUATION_PROMPT.format(
            task=task,
            worker_outputs=outputs_text,
        )

        result = self.run(prompt)
        self.trace.log(
            TraceStepType.SUCCESS,
            f"Synthesis complete ({len(result)} chars)",
        )

        return result