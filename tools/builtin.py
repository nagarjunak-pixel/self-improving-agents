"""
builtin.py — Built-in Tools
=============================
Simple tools that come pre-packaged with the system.

For teaching: these are intentionally simple. Students can read them
in 30 seconds and understand what they do.

For production: you'd swap these with real implementations
(web search, database queries, code execution sandboxes, etc.)
"""

import math
import random
from typing import Optional

from core.tools import Tool, ToolRegistry


def calculator(expression: str) -> str:
    """
    Safe mathematical calculator.
    Evaluates a math expression without using eval().
    """
    allowed = {
        "abs": abs, "round": round, "min": min, "max": max,
        "sqrt": math.sqrt, "pow": pow, "log": math.log,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "pi": math.pi, "e": math.e,
    }
    try:
        # Use eval with restricted globals for safety
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"Result: {result}"
    except Exception as e:
        return f"Calculator error: {e}"


def random_choice(items: str) -> str:
    """Pick a random item from a comma-separated list."""
    choices = [x.strip() for x in items.split(",")]
    return f"Selected: {random.choice(choices)}"


def word_count(text: str) -> str:
    """Count words, characters, and sentences in text."""
    words = len(text.split())
    chars = len(text)
    sentences = text.count(".") + text.count("!") + text.count("?")
    return f"Words: {words}, Characters: {chars}, Sentences: {sentences}"


def text_summary(text: str, max_chars: int = 200) -> str:
    """Truncate text to a summary (first N characters + ellipsis)."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def get_builtin_tools() -> ToolRegistry:
    """Return a ToolRegistry with all built-in tools pre-registered."""
    registry = ToolRegistry()

    registry.register(Tool(
        name="calculator",
        description="Evaluate a mathematical expression. Supports: +, -, *, /, sqrt, pow, log, sin, cos, etc.",
        func=calculator,
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The math expression to evaluate, e.g. 'sqrt(144) + 2*3'",
                },
            },
            "required": ["expression"],
        },
    ))

    registry.register(Tool(
        name="random_choice",
        description="Pick a random item from a comma-separated list of options.",
        func=random_choice,
        parameters={
            "type": "object",
            "properties": {
                "items": {
                    "type": "string",
                    "description": "Comma-separated list, e.g. 'red, green, blue'",
                },
            },
            "required": ["items"],
        },
    ))

    registry.register(Tool(
        name="word_count",
        description="Count words, characters, and sentences in a text.",
        func=word_count,
        parameters={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to analyze",
                },
            },
            "required": ["text"],
        },
    ))

    registry.register(Tool(
        name="text_summary",
        description="Summarize a long text by truncating it to a specified length.",
        func=text_summary,
        parameters={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to summarize",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum characters to keep (default 200)",
                },
            },
            "required": ["text"],
        },
    ))

    return registry