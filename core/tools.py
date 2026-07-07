"""
tools.py — Tool Registration System
====================================
Lets agents call external functions (search, calc, code exec, etc.)

WHY THIS EXISTS:
Agents need to DO things, not just think. This module:
  1. Defines how tools are described to the LLM (schema)
  2. Dispatches tool calls safely
  3. Handles errors gracefully (agents should fail soft, not crash)

This mirrors how production systems (OpenAI function calling, MCP) work —
but simplified so students can read every line.
"""

import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class Tool:
    """
    A single tool that an agent can call.

    Attributes:
        name: Unique identifier (e.g. "search_web")
        description: What the tool does (shown to the LLM)
        func: The actual Python function
        parameters: JSON schema describing the parameters
    """
    name: str
    description: str
    func: Callable[..., Any]
    parameters: dict = field(default_factory=dict)

    def call(self, **kwargs) -> str:
        """Execute the tool with given arguments. Returns string result."""
        try:
            result = self.func(**kwargs)
            # Always return string — LLMs eat strings
            if isinstance(result, (dict, list)):
                return json.dumps(result, indent=2, default=str)
            return str(result)
        except Exception as e:
            # Fail soft — return the error as a string, don't crash
            return f"TOOL_ERROR[{self.name}]: {type(e).__name__}: {e}"

    def to_schema(self) -> dict:
        """Convert to OpenAI function-calling schema format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """
    Central registry for all tools available to agents.

    Usage:
        registry = ToolRegistry()
        registry.register(Tool(name="search", ...))
        registry.register(Tool(name="calculate", ...))

        # Agent asks "what tools do I have?"
        schemas = registry.schemas()  # → list of tool schemas for the LLM

        # Agent decides to call a tool
        result = registry.execute("search", query="hello")
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def schemas(self) -> list[dict]:
        """All tool schemas — pass this to the LLM's tools parameter."""
        return [t.to_schema() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def execute(self, name: str, **kwargs) -> str:
        """Execute a tool by name. Returns string result or error message."""
        tool = self.get(name)
        if not tool:
            return f"TOOL_ERROR: Unknown tool '{name}'. Available: {self.names()}"
        return tool.call(**kwargs)

    def __len__(self):
        return len(self._tools)

    def __contains__(self, name: str):
        return name in self._tools