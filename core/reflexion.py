"""
reflexion.py — The Self-Improvement Engine
===========================================
This is what makes agents "self-improving" instead of just "retrying."

THE CONCEPT (from the Reflexion paper, Shinn et al. 2023):
============================================================
Normal agent fails → tries again → probably fails the same way.
Reflexive agent fails → REFLECTS on why it failed (verbal critique) →
tries again WITH that critique in context → does better.

The magic is in the REFLECTION step. Instead of blindly retrying,
the agent generates a natural language critique:
  "I failed because I called the search tool before checking if
   the database was connected. Next time, I should verify
   infrastructure before querying."

This critique is stored and injected into the next attempt's context.
It's like a student who, after failing an exam, writes down WHAT they
got wrong and WHY — then does better on the retake.

HOW IT WORKS HERE:
==================
1. Agent attempts a task
2. If it fails (or the Critic rejects it), trigger Reflexion
3. Ask the LLM: "You tried X. You got Y. Why did it fail? What should you do differently?"
4. Store the reflection as a LESSON in long-term memory
5. Retry the task with the lesson injected into the system prompt
6. Repeat up to N times

This is NOT fine-tuning. No weights change. The improvement is purely
from adding verbal self-critique to the context. That's what makes it
elegant — and teachable.
"""

from typing import Optional

from openai import OpenAI

from core.agent import BaseAgent
from core.memory import LongTermMemory
from core.trace import TraceLogger, TraceStepType


class ReflexionEngine:
    """
    Wraps any BaseAgent with self-improvement capability.

    Usage:
        engine = ReflexionEngine(agent=my_worker_agent, max_reflections=3)
        result = engine.run_with_reflexion("Solve this problem: ...")
    """

    def __init__(
        self,
        agent: BaseAgent,
        max_reflections: int = 3,
        model: str = "gpt-4o",
        client: Optional[OpenAI] = None,
    ):
        self.agent = agent
        self.max_reflections = max_reflections
        self.model = model
        self.client = client or OpenAI()
        self.reflection_trace = TraceLogger(agent_name=f"Reflexion-{agent.name}")

    def _generate_reflection(self, task: str, attempt: str, feedback: str) -> str:
        """
        Ask the LLM to reflect on a failed attempt.

        Args:
            task: What the agent was trying to do
            attempt: What the agent produced
            feedback: Why it wasn't good enough (from the Critic)

        Returns:
            A verbal reflection (string) — the "lesson learned"
        """
        prompt = f"""You are a reflective reasoning engine. An agent tried to complete a task but fell short.

## Task:
{task}

## Agent's Attempt:
{attempt}

## Feedback (why it wasn't good enough):
{feedback}

## Your Job:
Generate a CONCISE reflection (2-4 sentences) answering:
1. What went wrong?
2. What should the agent do differently next time?
3. What specific strategy change would help?

Be specific. "Try harder" is not a useful reflection. "Verify the database
connection before querying" IS a useful reflection.

Output ONLY the reflection, no preamble."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )
        return response.choices[0].message.content.strip()

    def run_with_reflexion(self, task: str, feedback_fn=None) -> dict:
        """
        Run the agent, and if it doesn't succeed, reflect and retry.

        Args:
            task: The task to complete
            feedback_fn: Optional function that evaluates the agent's output
                         and returns (is_good: bool, feedback: str)
                         If None, just returns the first attempt.

        Returns:
            {
                "final_answer": str,
                "attempts": list of (attempt, reflection) tuples,
                "succeeded": bool,
                "total_reflections": int,
            }
        """
        attempts = []
        current_task = task
        succeeded = False
        final_answer = ""

        for reflection_round in range(self.max_reflections + 1):
            self.reflection_trace.log(
                TraceStepType.REFLEXION,
                f"Round {reflection_round}/{self.max_reflections}",
            )

            # --- RUN THE AGENT ---
            answer = self.agent.run(current_task)
            self.reflection_trace.log(
                TraceStepType.OBSERVE,
                f"Agent produced {len(answer)} chars",
            )

            # --- EVALUATE ---
            if feedback_fn:
                is_good, feedback = feedback_fn(task, answer)
            else:
                # No feedback function → accept first attempt
                is_good, feedback = True, "No evaluator provided"

            if is_good:
                self.reflection_trace.log(
                    TraceStepType.SUCCESS,
                    f"Attempt {reflection_round + 1} succeeded!",
                )
                succeeded = True
                final_answer = answer
                attempts.append({"attempt": answer, "reflection": None, "success": True})
                break

            # --- REFLECT ---
            self.reflection_trace.log(
                TraceStepType.REFLECT,
                f"Attempt failed. Generating reflection...",
                data={"feedback": feedback[:200]},
            )

            reflection = self._generate_reflection(task, answer, feedback)

            self.reflection_trace.log(
                TraceStepType.REFLEXION,
                f"Reflection generated: {reflection[:100]}...",
            )

            # --- STORE THE LESSON ---
            if self.agent.long_term_memory:
                self.agent.long_term_memory.add_lesson(reflection)
                self.agent.long_term_memory.add_episode(
                    task=task, outcome=answer, reflection=reflection, success=False,
                )

            attempts.append({
                "attempt": answer,
                "reflection": reflection,
                "success": False,
            })

            # --- RETRY WITH REFLECTION INJECTED ---
            # Modify the task to include the lesson, so the agent sees it
            current_task = f"""{task}

## LESSON FROM PREVIOUS ATTEMPT:
{reflection}

Please try again, applying this lesson. Do NOT repeat the same mistakes."""

        if not succeeded:
            final_answer = attempts[-1]["attempt"] if attempts else ""
            self.reflection_trace.log(
                TraceStepType.ERROR,
                f"Failed after {self.max_reflections} reflections",
            )

        return {
            "final_answer": final_answer,
            "attempts": attempts,
            "succeeded": succeeded,
            "total_reflections": len(attempts) - 1 if succeeded else len(attempts),
        }