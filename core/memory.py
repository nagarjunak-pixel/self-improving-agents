"""
memory.py — Agent Memory System
================================
Short-term (conversation) + Long-term (persistent) memory for agents.

WHY THIS EXISTS:
Memory is what separates a chatbot from an agent. A chatbot forgets
everything between turns. An agent REMEMBERS.

This module implements three tiers:

1. SHORT-TERM (Working Memory)
   - Current conversation history
   - Lives in RAM, dies when the agent dies
   - Think: your brain's working memory right now

2. LONG-TERM (Episodic Memory)
   - Past experiences, successes, failures
   - Persists to disk (JSON)
   - Think: your journal of past events

3. REFLEXION MEMORY (Lessons Learned)
   - Verbal critiques from past failures
   - Used to improve future attempts
   - Think: "Last time I tried this, I forgot to check X. Don't forget this time."

For advanced students: this maps directly to how production systems
(Claude, GPT agents) manage context windows and persistent memory.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


@dataclass
class Message:
    """A single message in the conversation history."""
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict = field(default_factory=dict)  # tool name, call_id, etc.

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content, **self.metadata}

    def to_openai(self) -> dict:
        """Convert to OpenAI message format."""
        return {"role": self.role, "content": self.content}


class ShortTermMemory:
    """
    Working memory — the conversation the agent is currently having.
    Bounded by a max length to simulate context window limits.

    When the buffer is full, oldest messages are dropped (FIFO),
    but the system message is ALWAYS kept.
    """

    def __init__(self, system_prompt: str, max_messages: int = 50):
        self.messages: list[Message] = [Message(role="system", content=system_prompt)]
        self.max_messages = max_messages

    def add(self, role: str, content: str, **metadata) -> None:
        """Add a message to working memory."""
        self.messages.append(Message(role=role, content=content, metadata=metadata))
        self._trim()

    def _trim(self) -> None:
        """Drop oldest messages if over limit (keep system message)."""
        if len(self.messages) <= self.max_messages:
            return
        system = self.messages[0]
        rest = self.messages[1:]
        rest = rest[-(self.max_messages - 1):]
        self.messages = [system] + rest

    def to_openai(self) -> list[dict]:
        """Convert to OpenAI message format for API calls."""
        return [m.to_openai() for m in self.messages]

    def clear(self, new_system: Optional[str] = None) -> None:
        """Wipe working memory. Optionally set a new system prompt."""
        if new_system:
            self.messages = [Message(role="system", content=new_system)]
        else:
            self.messages = []

    def __len__(self):
        return len(self.messages)

    def __repr__(self):
        return f"<ShortTermMemory: {len(self.messages)} messages>"


class LongTermMemory:
    """
    Persistent memory stored on disk.
    Episodes = past experiences (task + outcome + reflection).
    Lessons = distilled insights from reflexion.

    Think of episodes as a journal and lessons as a list of
    "things I've learned the hard way."
    """

    def __init__(self, path: str = "memory/long_term.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.episodes: list[dict] = []
        self.lessons: list[str] = []
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            data = json.loads(self.path.read_text())
            self.episodes = data.get("episodes", [])
            self.lessons = data.get("lessons", [])

    def _save(self) -> None:
        self.path.write_text(json.dumps(
            {"episodes": self.episodes, "lessons": self.lessons},
            indent=2, default=str,
        ))

    def add_episode(self, task: str, outcome: str, reflection: str, success: bool) -> None:
        """Record a past experience."""
        episode = {
            "timestamp": datetime.now().isoformat(),
            "task": task,
            "outcome": outcome,
            "reflection": reflection,
            "success": success,
        }
        self.episodes.append(episode)
        self._save()

    def add_lesson(self, lesson: str) -> None:
        """Record a lesson learned through reflexion."""
        if lesson not in self.lessons:  # Dedupe
            self.lessons.append(lesson)
            self._save()

    def get_lessons(self, limit: int = 10) -> list[str]:
        """Return the N most recent lessons (for injecting into agent prompts)."""
        return self.lessons[-limit:]

    def get_relevant_episodes(self, task: str, limit: int = 3) -> list[dict]:
        """
        Simple keyword-based retrieval of past episodes.
        In production, you'd use embeddings here. For teaching,
        keyword overlap is clear and honest.
        """
        task_words = set(task.lower().split())
        scored = []
        for ep in self.episodes:
            ep_words = set(ep["task"].lower().split())
            overlap = len(task_words & ep_words)
            scored.append((overlap, ep))
        scored.sort(key=lambda x: -x[0])
        return [ep for _, ep in scored[:limit] if scored]

    def __repr__(self):
        return f"<LongTermMemory: {len(self.episodes)} episodes, {len(self.lessons)} lessons>"