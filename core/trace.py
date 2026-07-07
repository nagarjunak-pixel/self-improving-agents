"""
trace.py — Visual Trace Logger
================================
Logs every step of an agent's reasoning in real-time with rich formatting.

WHY THIS EXISTS:
In production agents, you often can't see "what happened inside."
This trace system makes the reasoning loop VISIBLE — critical for:
  1. Debugging (why did the agent fail?)
  2. Teaching (students see the think → act → observe loop)
  3. Trust (users can audit the agent's decisions)

Think of it as a flight recorder for agents.
"""

import json
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.json import JSON

console = Console()


class TraceStepType(Enum):
    """Every type of step an agent can take."""
    THINK = "🧠 THINK"          # Agent is reasoning about what to do
    ACT = "🔧 ACT"              # Agent is calling a tool
    OBSERVE = "👁️ OBSERVE"      # Agent is processing tool output
    REFLECT = "🪞 REFLECT"      # Agent is reflecting on a failure
    PLAN = "📋 PLAN"            # Agent is breaking down a task
    DECIDE = "⚖️ DECIDE"        # Agent is making a decision
    ERROR = "❌ ERROR"          # Something went wrong
    SUCCESS = "✅ SUCCESS"      # Something worked
    REFLEXION = "🔄 REFLEXION"  # Self-improvement cycle triggered


class TraceLogger:
    """
    Records and displays every step of agent reasoning.

    Usage:
        trace = TraceLogger(agent_name="Planner")
        trace.log(TraceStepType.THINK, "I need to break this task into 3 parts...")
        trace.log(TraceStepType.ACT, "Calling tool: search", data={"query": "..."})
        trace.log(TraceStepType.OBSERVE, "Got 5 results back")
    """

    def __init__(self, agent_name: str, log_dir: str = "trace"):
        self.agent_name = agent_name
        self.steps: list[dict] = []
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = time.time()

    def log(
        self,
        step_type: TraceStepType,
        content: str,
        data: Optional[dict] = None,
        display: bool = True,
    ) -> dict:
        """
        Record a single trace step.

        Args:
            step_type: What kind of step is this?
            content: Human-readable description
            data: Optional structured data (tool args, results, etc.)
            display: Show in console? (False for silent internal steps)

        Returns:
            The recorded step dict (for chaining)
        """
        step = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_s": round(time.time() - self.start_time, 3),
            "agent": self.agent_name,
            "type": step_type.value,
            "content": content,
            "data": data or {},
        }
        self.steps.append(step)

        if display:
            self._render(step)

        return step

    def _render(self, step: dict):
        """Pretty-print a step to the console using rich."""
        title = f"[bold cyan]{step['agent']}[/] · {step['type']} · [dim]{step['elapsed_s']}s[/]"

        body = step["content"]
        if step["data"]:
            body += f"\n\n[dim]Data:[/]\n{json.dumps(step['data'], indent=2, default=str)[:500]}"

        color_map = {
            "THINK": "blue",
            "ACT": "yellow",
            "OBSERVE": "green",
            "REFLECT": "magenta",
            "PLAN": "cyan",
            "DECIDE": "yellow",
            "ERROR": "red",
            "SUCCESS": "green",
            "REFLEXION": "magenta",
        }

        color = "white"
        for key, c in color_map.items():
            if key in step["type"]:
                color = c
                break

        console.print(Panel(body, title=title, border_style=color, padding=(0, 1)))

    def save(self, filename: Optional[str] = None) -> Path:
        """Save the full trace to a JSON file for later analysis."""
        if not filename:
            filename = f"{self.agent_name.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path = self.log_dir / filename
        path.write_text(json.dumps(self.steps, indent=2, default=str))
        return path

    def summary(self) -> str:
        """Return a one-line summary of the trace."""
        counts: dict[str, int] = {}
        for s in self.steps:
            counts[s["type"]] = counts.get(s["type"], 0) + 1
        parts = [f"{v}× {k}" for k, v in counts.items()]
        return f"{self.agent_name}: {', '.join(parts)} in {self.steps[-1]['elapsed_s'] if self.steps else 0}s"