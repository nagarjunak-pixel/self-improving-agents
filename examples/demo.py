#!/usr/bin/env python3
"""
demo.py — Full System Demo
============================
Run this to see the entire self-improving multi-agent system in action.

Usage:
    python examples/demo.py
    python examples/demo.py --task "Design a scalable architecture for a real-time chat application"
    python examples/demo.py --task "Compare 3 ML frameworks for production recommendation systems"
    python examples/demo.py --model gpt-4o-mini  # cheaper model

PREREQUISITES:
    pip install -r requirements.txt
    export OPENAI_API_KEY="sk-..."
"""

import sys
import os
import json
import argparse

# Make the package importable when running from examples/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Self-Improving Multi-Agent System Demo")
    parser.add_argument(
        "--task",
        type=str,
        default="Design a REST API for a todo application with user authentication, including database schema, endpoint list, and security considerations.",
        help="The task for the agent system to work on",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model to use (gpt-4o, gpt-4o-mini, etc.)",
    )
    parser.add_argument(
        "--max-reflections",
        type=int,
        default=2,
        help="Maximum reflexion rounds per subtask",
    )
    parser.add_argument(
        "--no-tools",
        action="store_true",
        help="Disable built-in tools (pure reasoning mode)",
    )
    args = parser.parse_args()

    # --- HEADER ---
    console.print(Panel.fit(
        f"[bold cyan]🧠 Self-Improving Multi-Agent System[/]\n\n"
        f"[dim]Model: {args.model}\n"
        f"Max Reflexions: {args.max_reflections}\n"
        f"Tools: {'disabled' if args.no_tools else 'builtin (calculator, word_count, etc.)'}[/]",
        border_style="cyan",
    ))

    # --- CHECK API KEY ---
    if not os.environ.get("OPENAI_API_KEY"):
        console.print("[red]ERROR: OPENAI_API_KEY not set. Export it first:[/]")
        console.print("  export OPENAI_API_KEY=\"sk-...\"")
        sys.exit(1)

    # --- IMPORTS (deferred until after API key check) ---
    from core.memory import LongTermMemory
    from core.tools import ToolRegistry
    from agents.orchestrator import Orchestrator
    from tools.builtin import get_builtin_tools

    # --- SETUP ---
    long_term_memory = LongTermMemory(path="memory/long_term.json")
    tools = ToolRegistry() if args.no_tools else get_builtin_tools()

    orchestrator = Orchestrator(
        model=args.model,
        tools=tools,
        long_term_memory=long_term_memory,
        max_reflections=args.max_reflections,
    )

    # --- RUN ---
    console.print(Panel(f"[bold yellow]TASK:[/]\n{args.task}", border_style="yellow"))

    result = orchestrator.run(args.task)

    # --- RESULTS ---
    console.print("\n")
    console.print(Panel.fit("[bold green]✅ RESULTS[/]", border_style="green"))

    # Summary table
    table = Table(title="Subtask Summary")
    table.add_column("#", style="cyan")
    table.add_column("Subtask", style="white")
    table.add_column("Reflections", style="magenta")
    table.add_column("Status", style="green")

    for i, w in enumerate(result["worker_outputs"], 1):
        status = "✅ Approved" if w["approved"] else "❌ Failed"
        table.add_row(
            str(i),
            w["subtask"][:60] + ("..." if len(w["subtask"]) > 60 else ""),
            str(w["total_reflections"]),
            status,
        )

    console.print(table)

    # Final answer
    console.print(Panel(
        result["final_answer"],
        title="[bold green]Final Answer[/]",
        border_style="green",
    ))

    # Stats
    console.print(f"\n[dim]Total time: {result['total_time_s']}s[/]")
    console.print(f"[dim]Long-term memory: {len(long_term_memory.episodes)} episodes, {len(long_term_memory.lessons)} lessons[/]")
    console.print(f"[dim]Traces saved to: trace/[/]")

    # If reflexion happened, show the improvement
    for w in result["worker_outputs"]:
        if w["total_reflections"] > 0:
            console.print(f"\n[magenta]📝 Reflexion for: {w['subtask'][:50]}...[/]")
            for j, attempt in enumerate(w["attempts"]):
                if attempt["reflection"]:
                    console.print(f"  [dim]Round {j+1} reflection:[/]")
                    console.print(f"  [italic]{attempt['reflection'][:200]}[/]")


if __name__ == "__main__":
    main()