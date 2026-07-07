"""
orchestrator.py — Multi-Agent Orchestrator
===========================================
The conductor of the entire system.

WHY THIS EXISTS:
This ties everything together. The orchestrator:
  1. Takes a task
  2. Asks the PLANNER to decompose it
  3. Spins up WORKERS for each subtask
  4. Has the CRITIC evaluate each worker's output
  5. If the Critic rejects → triggers REFLEXION (retry with self-improvement)
  6. Sends all approved outputs to the JUDGE for final synthesis
  7. Returns the final answer

This is the same pattern used by production systems like:
  - AutoGPT (plan → execute → review → retry)
  - Devin (decompose → code → test → fix)
  - SWE-Agent (understand → edit → run tests → debug)

THE FLOW:
=========
    Task → Planner → [Subtask 1, Subtask 2, ...]
                        ↓
    For each subtask:
        Worker → produces output
        Critic → evaluates output
        If rejected:
            Reflexion → Worker retries with lesson
            Critic → re-evaluates
        If approved: → to Judge
                        ↓
    Judge synthesizes all approved outputs → Final Answer

KEY CONCEPT FOR STUDENTS:
The Orchestrator is itself an agent — but instead of using TOOLS,
it uses OTHER AGENTS. Agents calling agents. This is "meta-agentic"
behavior, and it's where the field is heading.
"""

import time
from typing import Optional

from openai import OpenAI

from core.agent import BaseAgent
from core.memory import LongTermMemory
from core.tools import ToolRegistry
from core.reflexion import ReflexionEngine
from core.trace import TraceLogger, TraceStepType

from agents.planner import PlannerAgent
from agents.worker import WorkerAgent
from agents.critic import CriticAgent
from agents.judge import JudgeAgent


class Orchestrator:
    """
    Coordinates the full multi-agent pipeline:
    Planner → Workers → Critic → Reflexion → Judge
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        tools: Optional[ToolRegistry] = None,
        long_term_memory: Optional[LongTermMemory] = None,
        max_reflections: int = 2,
        client: Optional[OpenAI] = None,
    ):
        self.model = model
        self.tools = tools or ToolRegistry()
        self.long_term_memory = long_term_memory or LongTermMemory()
        self.max_reflections = max_reflections
        self.client = client or OpenAI()
        self.trace = TraceLogger(agent_name="Orchestrator")

        # Create all agents
        self.planner = PlannerAgent(model=model, client=self.client)
        self.critic = CriticAgent(model=model, client=self.client)
        self.judge = JudgeAgent(model=model, client=self.client)

    def run(self, task: str) -> dict:
        """
        Run the full multi-agent pipeline.

        Returns:
            {
                "task": str,
                "subtasks": list,
                "worker_outputs": list of {subtask, output, attempts, approved},
                "final_answer": str,
                "total_time_s": float,
            }
        """
        start = time.time()
        self.trace.log(TraceStepType.PLAN, f"Starting orchestration: {task}")

        # --- STEP 1: PLAN ---
        self.trace.log(TraceStepType.THINK, "Asking Planner to decompose task...")
        subtasks = self.planner.plan(task)

        # --- STEP 2: EXECUTE EACH SUBTASK ---
        worker_outputs = []
        for subtask in subtasks:
            subtask_desc = subtask.get("task", subtask.get("description", str(subtask)))
            self.trace.log(
                TraceStepType.PLAN,
                f"Processing subtask {subtask.get('id', '?')}: {subtask_desc[:80]}...",
            )

            # Create a fresh worker for this subtask
            worker = WorkerAgent(
                name=f"Worker-{subtask.get('id', '?')}",
                model=self.model,
                tools=self.tools,
                long_term_memory=self.long_term_memory,
                client=self.client,
            )

            # Wrap worker with Reflexion + Critic evaluation
            reflexion_engine = ReflexionEngine(
                agent=worker,
                max_reflections=self.max_reflections,
                model=self.model,
                client=self.client,
            )

            # The Critic serves as the feedback function for Reflexion
            def critic_feedback(original_task: str, attempt: str) -> tuple[bool, str]:
                return self.critic.evaluate(original_task, attempt)

            # Run with self-improvement
            result = reflexion_engine.run_with_reflexion(
                task=subtask_desc,
                feedback_fn=critic_feedback,
            )

            worker_outputs.append({
                "subtask": subtask_desc,
                "output": result["final_answer"],
                "attempts": result["attempts"],
                "approved": result["succeeded"],
                "total_reflections": result["total_reflections"],
            })

        # --- STEP 3: SYNTHESIZE ---
        self.trace.log(
            TraceStepType.THINK,
            f"Sending {len(worker_outputs)} outputs to Judge for synthesis...",
        )
        final_answer = self.judge.synthesize(task, worker_outputs)

        elapsed = round(time.time() - start, 2)
        self.trace.log(
            TraceStepType.SUCCESS,
            f"Orchestration complete in {elapsed}s",
        )

        # Save traces
        self.trace.save()
        self.planner.trace.save()
        self.critic.trace.save()
        self.judge.trace.save()

        return {
            "task": task,
            "subtasks": subtasks,
            "worker_outputs": worker_outputs,
            "final_answer": final_answer,
            "total_time_s": elapsed,
        }