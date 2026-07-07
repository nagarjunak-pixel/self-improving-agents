# 🧠 Self-Improving Multi-Agent System

A teaching framework that demonstrates **Reflexion**, **Multi-Agent Orchestration**, **Memory**, and **Tool Use** — the cutting edge of agentic AI.

## What This Teaches

| Concept | Where to Look |
|---|---|
| ReAct Reasoning Loop | `core/agent.py` — `run()` method |
| Reflexion (Self-Improvement) | `core/reflexion.py` |
| Short + Long-Term Memory | `core/memory.py` |
| Tool Registration & Calling | `core/tools.py` |
| Multi-Agent Orchestration | `agents/orchestrator.py` |
| Adversarial Critic Agent | `agents/critic.py` |
| Agent-as-a-Judge | `agents/judge.py` |
| Visual Trace Logging | `core/trace.py` |

## Architecture

```
┌─────────────────────────────────────┐
│         TASK INPUT                  │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│   PLANNER AGENT                     │
│   Breaks task into subtasks         │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│   WORKER AGENTS (parallel)          │
│   Each tackles a subtask            │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│   CRITIC AGENT (adversarial)        │
│   Finds flaws, challenges outputs   │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│   REFLEXION LAYER                   │
│   Reflects on failures,             │
│   generates verbal critique,        │
│   updates strategy, retries         │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│   JUDGE AGENT                       │
│   Synthesizes final answer + scores │
└─────────────────────────────────────┘
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set your API key
export OPENAI_API_KEY="sk-..."

# Run the demo
python examples/demo.py

# Run with a custom task
python examples/demo.py --task "Design a scalable architecture for a real-time chat application"
```

## Design Principles

1. **Readable** — Every file is small enough to read in one sitting
2. **Honest** — No magic. Every decision is commented.
3. **Extensible** — Add a new agent? Subclass `BaseAgent`. Add a tool? Register it.
4. **Observable** — Full trace of every thought, action, and reflection.

## License
MIT — Use it, teach it, break it, learn from it.