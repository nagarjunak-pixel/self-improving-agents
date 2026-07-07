"""
critic.py — The Critic Agent (Adversarial Reasoning)
=====================================================
Finds flaws in other agents' work.

WHY THIS EXISTS:
This is ADVERSARIAL REASONING — one of the most powerful patterns in
modern agentic AI. Instead of trusting the first answer, a separate
agent challenges it.

WHY IT WORKS (the science):
Research on "Multi-Agent Debate" (Du et al. 2023) shows that having
multiple agents argue produces better results than a single agent
reasoning alone. The Critic forces the Worker to justify its work,
catching errors the Worker would miss.

THE PATTERN:
  Worker: "X is the answer because A, B, C."
  Critic: "But B is wrong because [reason]. Also, you didn't consider D."
  Worker (with Reflexion): "OK, B was wrong. The real answer is X' because A, D, and E."

In production systems, this is how "Agent-as-a-Judge" works — you
have an agent evaluate another agent's output.

KEY CONCEPT FOR STUDENTS:
This is NOT the same as having a human review. The Critic is an LLM
with a specific adversarial prompt. Its ONLY job is to find problems.
This separation of concerns (one agent creates, another critiques)
is fundamental to multi-agent architecture.
"""

from typing import Optional

from openai import OpenAI

from core.agent import BaseAgent
from core.trace import TraceLogger, TraceStepType

CRITIC_PROMPT = """You are a CRITIC agent in a multi-agent system.

Your job: Find flaws in the work produced by other agents.

You are NOT trying to be nice. You are NOT trying to improve the work.
You are trying to FIND PROBLEMS. Be ruthless but fair.

For each piece of work, evaluate:
1. **Correctness**: Are there factual errors or logical flaws?
2. **Completeness**: Did it miss anything important?
3. **Specificity**: Is it vague when it should be concrete?
4. **Assumptions**: Are there unstated assumptions that might be wrong?
5. **Counter-arguments**: What would a skeptic say?

Output format:
- If the work is acceptable: "APPROVED: <brief note on what's good>"
- If the work needs improvement: "REJECTED: <specific list of problems>"

Be specific. "Too vague" is unhelpful. "Section 2 doesn't specify which database to use, and the scalability claims lack evidence" is helpful."""


class CriticAgent(BaseAgent):
    """Evaluates other agents' work adversarially."""

    def __init__(self, model: str = "gpt-4o", client: Optional[OpenAI] = None, **kwargs):
        super().__init__(
            name="Critic",
            system_prompt=CRITIC_PROMPT,
            model=model,
            tools=None,
            max_iterations=1,  # One shot — critique and respond
            client=client,
            **kwargs,
        )

    def evaluate(self, task: str, work: str) -> tuple[bool, str]:
        """
        Evaluate a piece of work.

        Args:
            task: What the original task was
            work: What the worker agent produced

        Returns:
            (is_approved: bool, feedback: str)
        """
        self.trace.log(
            TraceStepType.DECIDE,
            f"Evaluating work for: {task[:100]}...",
        )

        prompt = f"""## Original Task:
{task}

## Work to Evaluate:
{work}

Now evaluate it."""

        result = self.run(prompt)

        is_approved = result.strip().upper().startswith("APPROVED")
        self.trace.log(
            TraceStepType.SUCCESS if is_approved else TraceStepType.ERROR,
            f"{'Approved' if is_approved else 'Rejected'}",
            data={"feedback": result[:300]},
        )
        return is_approved, result