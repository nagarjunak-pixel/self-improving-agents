"""
agent-api.py — Netlify Serverless Function
==========================================
The backend API that runs the multi-agent system and streams traces
back to the frontend via Server-Sent Events (SSE).

WHY SERVERLESS + SSE:
- Netlify functions are serverless (spin up, handle request, spin down)
- SSE lets us stream agent traces in real-time (student sees each step as it happens)
- No need for a persistent server — cheaper, simpler, perfect for a teaching demo

ARCHITECTURE:
    Browser → POST /netlify/functions/agent-api → Netlify spins up Python function
    → Function runs the multi-agent system
    → Streams trace events back as SSE
    → Browser renders each step in real-time
    → Function returns final result
    → Netlify spins down the function

KEY CONCEPT FOR STUDENTS:
This is how production AI apps work — a frontend talks to a backend
that runs the agent, and the agent's intermediate steps are streamed
back for real-time observability. Same pattern as ChatGPT's streaming.
"""

import json
import sys
import os
import traceback

# Add the project root to Python path so we can import our modules
# Netlify puts the function in /netlify/functions/ but our code is at root
FUNCTION_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(FUNCTION_DIR))
sys.path.insert(0, PROJECT_ROOT)

# We need to monkey-patch the trace system to stream events instead of
# just printing to console. This is the bridge between the backend agent
# system and the frontend real-time display.

from openai import OpenAI
from core.memory import LongTermMemory
from core.tools import ToolRegistry
from core.trace import TraceLogger, TraceStepType
from agents.orchestrator import Orchestrator
from tools.builtin import get_builtin_tools


class StreamingTraceLogger(TraceLogger):
    """
    A trace logger that sends events to a callback (which the Netlify
    function converts to SSE events for the browser).

    This is the same TraceLogger, but instead of just printing to console,
    it ALSO sends each step to the browser in real-time.

    WHY THIS MATTERS:
    In production, observability is critical. You want to see what the
    agent is doing AS IT HAPPENS, not after it's done. This pattern
    (streaming traces) is how ChatGPT, Claude, and others show you
    tokens as they're generated.
    """

    def __init__(self, agent_name: str, stream_callback=None, **kwargs):
        super().__init__(agent_name, **kwargs)
        self.stream_callback = stream_callback

    def log(self, step_type, content, data=None, display=True):
        step = super().log(step_type, content, data, display=False)

        # Stream to browser
        if self.stream_callback:
            self.stream_callback({
                "type": "trace",
                "step_type": step_type.value.split()[-1].lower(),  # "THINK" → "think"
                "agent": self.agent_name,
                "content": content,
                "data": data or {},
                "elapsed": step["elapsed_s"],
            })

        return step


def handler(event, context):
    """
    Netlify function handler.

    Receives: POST body with { task, model, max_reflections, api_key }
    Returns: SSE stream of trace events + final result
    """
    try:
        # Parse the request
        if event.get("httpMethod") != "POST":
            return {
                "statusCode": 405,
                "body": json.dumps({"error": "Method not allowed. Use POST."})
            }

        body = json.loads(event.get("body", "{}"))
        task = body.get("task", "").strip()
        model = body.get("model", "gpt-4o")
        max_reflections = body.get("max_reflections", 2)
        api_key = body.get("api_key", "")

        if not task:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "No task provided"})
            }
        if not api_key:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "No API key provided"})
            }

        # Collect trace events for SSE
        trace_events = []

        def stream_callback(event_data):
            trace_events.append(event_data)

        # Monkey-patch TraceLogger to use our streaming version
        original_init = TraceLogger.__init__
        def patched_init(self, agent_name, **kwargs):
            StreamingTraceLogger.__init__(self, agent_name, stream_callback, **kwargs)
        TraceLogger.__init__ = patched_init

        # Run the orchestrator
        client = OpenAI(api_key=api_key)
        long_term_memory = LongTermMemory(path="/tmp/long_term.json")
        tools = get_builtin_tools()

        orchestrator = Orchestrator(
            model=model,
            tools=tools,
            long_term_memory=long_term_memory,
            max_reflections=max_reflections,
            client=client,
        )

        result = orchestrator.run(task)

        # Restore original TraceLogger
        TraceLogger.__init__ = original_init

        # Build SSE response
        sse_lines = []
        for evt in trace_events:
            sse_lines.append(f"data: {json.dumps(evt)}\n")

        # Add final result
        sse_lines.append(f"data: {json.dumps({'type': 'result', 'data': result})}\n")

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
            "body": "\n".join(sse_lines),
            "isBase64Encoded": False,
        }

    except Exception as e:
        traceback.print_exc()
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "type": "error",
                "message": f"Internal error: {str(e)}"
            })
        }