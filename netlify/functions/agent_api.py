"""
Netlify Python Function: agent-api
Runs the multi-agent system and returns SSE stream.
"""

import json
import sys
import os
import traceback

# Add project root to path
FUNCTION_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(FUNCTION_DIR))
sys.path.insert(0, PROJECT_ROOT)

from openai import OpenAI
from core.memory import LongTermMemory
from core.tools import ToolRegistry
from core.trace import TraceLogger, TraceStepType
from agents.orchestrator import Orchestrator
from tools.builtin import get_builtin_tools


class StreamingTraceLogger(TraceLogger):
    def __init__(self, agent_name, stream_callback=None, **kwargs):
        super().__init__(agent_name, **kwargs)
        self.stream_callback = stream_callback

    def log(self, step_type, content, data=None, display=True):
        step = super().log(step_type, content, data, display=False)
        if self.stream_callback:
            self.stream_callback({
                "type": "trace",
                "step_type": step_type.value.split()[-1].lower(),
                "agent": self.agent_name,
                "content": content,
                "data": data or {},
                "elapsed": step["elapsed_s"],
            })
        return step


def handler(event, context):
    try:
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
            return {"statusCode": 400, "body": json.dumps({"error": "No task provided"})}
        if not api_key:
            return {"statusCode": 400, "body": json.dumps({"error": "No API key provided"})}

        trace_events = []

        def stream_callback(event_data):
            trace_events.append(event_data)

        original_init = TraceLogger.__init__
        def patched_init(self, agent_name, **kwargs):
            StreamingTraceLogger.__init__(self, agent_name, stream_callback, **kwargs)
        TraceLogger.__init__ = patched_init

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

        TraceLogger.__init__ = original_init

        sse_lines = []
        for evt in trace_events:
            sse_lines.append(f"data: {json.dumps(evt)}\n")
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
            "body": json.dumps({"type": "error", "message": f"Internal error: {str(e)}"})
        }