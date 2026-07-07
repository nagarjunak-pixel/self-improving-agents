# 🎓 TRAINER'S GUIDE — Self-Improving Multi-Agent System

*A complete walkthrough for instructors. Read this before teaching.*

---

## HOW TO USE THIS GUIDE

1. Read this top-to-bottom before the session
2. Each section = one file in the codebase
3. Each section has:
   - **THE CONCEPT** — what is this and why does it exist in AI?
   - **THE PROBLEM** — what problem does this solve?
   - **THE ANALOGY** — how to explain it to students
   - **THE CODE WALKTHROUGH** — key lines to point at
   - **TEACHING TIP** — how to make it click
   - **DISCUSSION QUESTIONS** — to spark class conversation

---

## LAYER 1: THE REASONING LOOP (`core/agent.py`)

### THE CONCEPT
Every AI agent — ChatGPT, Claude, AutoGPT, Devin — runs the SAME fundamental loop:

```
THINK → ACT → OBSERVE → THINK → ACT → OBSERVE → ... → DONE
```

This is called **ReAct** (Reasoning + Acting), from the paper by Yao et al. (2022). It's the single most important pattern in agentic AI. If your students understand this loop, they understand 90% of how all agents work.

### THE PROBLEM
A plain LLM just generates text. It can't DO anything. It can't search the web, run code, or check a database. The LLM is a brain in a jar.

The ReAct loop solves this by giving the LLM a way to interact with the world:
- THINK: "What do I need to do next?"
- ACT: "I'll call this tool with these arguments"
- OBSERVE: "The tool returned this result"
- THINK: "Based on that result, what should I do now?"

### THE ANALOGY
> Imagine a detective at a crime scene. They can't just sit and think — they need to gather evidence. So they:
> 1. THINK: "I need to check the security camera footage"
> 2. ACT: Walk to the security room, request footage
> 3. OBSERVE: Watch the footage, see a suspect
> 4. THINK: "Now I need to find out who this person is"
> 5. ACT: Run facial recognition
> 6. OBSERVE: Match found — John Smith
> 7. THINK: "I now have enough to write my report"
>
> The detective IS a ReAct loop. The brain (LLM) decides what to do, the hands (tools) do it, the eyes (observation) feed info back.

### THE CODE WALKTHROUGH

```python
# THE LOOP — this is it. The entire agent in ~20 lines.
for i in range(self.max_iterations):
    # 1. CALL THE LLM — "What should I do?"
    response = self.client.chat.completions.create(...)
    msg = response.choices[0].message

    # 2. DID THE LLM WANT TO CALL A TOOL?
    if msg.tool_calls:
        for tool_call in msg.tool_calls:
            # 3. EXECUTE THE TOOL — hands do the work
            result = self.tools.execute(tool_name, **args)
            # 4. FEED RESULT BACK — eyes see what happened
            self.memory.add("tool", result)
        # 5. LOOP BACK — think again with new info
        continue

    # 6. NO TOOL CALL → the agent is done, has a final answer
    answer = msg.content
    return answer
```

**Lines to highlight to students:**
- `for i in range(self.max_iterations)` — the loop. Agents don't run forever. There's always a limit.
- `if msg.tool_calls:` — this is the branch that makes an agent an AGENT, not just a chatbot
- `self.memory.add("tool", result)` — this is the observe step. Without memory, the agent has amnesia.
- `continue` — this sends the agent back to the top to think again with new info

### TEACHING TIP
Open `core/agent.py` on screen. Delete the comments. Ask students: "Can you identify the think, act, observe steps?" Let them find it themselves before you explain it.

### DISCUSSION QUESTIONS
1. What happens if `max_iterations` is too low? (Agent gives up before solving the problem)
2. What happens if it's too high? (Agent loops forever, burning money)
3. Could an agent call itself recursively? (Yes — this is multi-agent orchestration!)
4. What's the difference between this and a while loop that just calls the LLM repeatedly? (The tool-calling branch — that's what makes it an agent)

---

## LAYER 2: REFLEXION (`core/reflexion.py`)

### THE CONCEPT
From the paper "Reflexion: Language Agents with Verbal Reinforcement Learning" (Shinn et al., 2023).

Normal agent fails → retries → probably fails the SAME way.
Reflexive agent fails → REFLECTS on why → retries differently → does better.

The key insight: **the agent doesn't just retry — it generates a verbal critique of its own failure and uses that critique as context for the next attempt.** No weights change. No fine-tuning. Pure prompt-level self-improvement.

### THE PROBLEM
LLMs make the same mistakes repeatedly because they don't learn from single-session failures. If an agent tries to solve a coding problem, fails because it forgot to import a library, and you just say "try again" — it'll probably forget the import again.

But if you say: "You tried X. You failed because you forgot to import the library. Try again and remember to import it." — now it'll succeed.

Reflexion automates this. The agent critiques ITSELF.

### THE ANALOGY
> Two students fail a math test.
>
> Student A: Takes the test again. Probably makes the same mistakes.
>
> Student B: After failing, writes down: "I forgot to carry the 1 in long division. I also confused sine and cosine. Next time: double-check carrying in division, and draw the triangle for trig functions."
>
> Student B takes the test again — and does much better.
>
> Student B is using Reflexion. The verbal critique is the "lesson learned."

### THE CODE WALKTHROUGH

```python
# THE REFLEXION LOOP
for reflection_round in range(self.max_reflections + 1):
    # 1. Run the agent normally
    answer = self.agent.run(current_task)

    # 2. Evaluate the answer (using the Critic)
    is_good, feedback = feedback_fn(task, answer)

    # 3. If good → done!
    if is_good:
        break

    # 4. If bad → GENERATE A REFLECTION
    reflection = self._generate_reflection(task, answer, feedback)
    # ^ This asks the LLM: "You tried X. You got Y. Why did it fail?
    #   What should you do differently?"

    # 5. Store the lesson in long-term memory
    self.agent.long_term_memory.add_lesson(reflection)

    # 6. Retry WITH the lesson injected into the task
    current_task = f"""{task}
    ## LESSON FROM PREVIOUS ATTEMPT:
    {reflection}
    Please try again, applying this lesson."""
```

**Lines to highlight:**
- `reflection = self._generate_reflection(...)` — this is the magic moment. The agent talks to itself about its own failure.
- `self.agent.long_term_memory.add_lesson(reflection)` — the lesson is PERMANENT. Future runs of the agent will see this lesson in their system prompt.
- `current_task = f"...\n## LESSON:\n{reflection}\n..."` — the lesson is injected into the NEXT attempt. This is how the agent "learns."

### TEACHING TIP
Run the demo TWICE on the same task:
1. First time: let it fail and generate reflections
2. Second time: the lessons are in long-term memory — it should succeed faster
3. Show students `memory/long_term.json` between runs — they can SEE the lessons stored on disk

### DISCUSSION QUESTIONS
1. Is this "real" learning? (No — no weights change. It's in-context learning. But it works!)
2. How is this different from fine-tuning? (Fine-tuning changes model weights permanently. Reflexion adds text to context. Cheaper, faster, but temporary.)
3. What happens if the reflection is wrong? (The agent might fail WORSE next time. This is a real problem in production systems.)
4. Could two agents reflect on each other? (Yes! This is multi-agent debate.)

---

## LAYER 3: MEMORY (`core/memory.py`)

### THE CONCEPT
Three tiers of memory, mirroring human cognition:

1. **Short-Term (Working) Memory** — the current conversation. Like your brain holding a phone number. Limited capacity. Dies when the session ends.
2. **Long-Term (Episodic) Memory** — past experiences stored on disk. Like a journal. Persists across sessions.
3. **Reflexion Memory (Lessons)** — distilled insights from failures. Like "things I've learned the hard way."

### THE PROBLEM
Without memory, every conversation starts from zero. The agent doesn't remember what it did yesterday, what mistakes it made, or what it learned.

This is the difference between ChatGPT (no persistent memory by default) and a true agent (remembers across sessions).

### THE ANALOGY
> **Short-term memory** = Your brain right now. You're holding this conversation in your head. If someone distracts you, you might lose it.
>
> **Long-term memory** = Your journal. You write down what happened to you. You can re-read it years later.
>
> **Lessons learned** = A sticky note on your monitor that says "Always check if the server is running before deploying." It's the distilled wisdom from past mistakes.

### THE CODE WALKTHROUGH

```python
# Short-term: bounded buffer with FIFO eviction
class ShortTermMemory:
    def _trim(self):
        if len(self.messages) <= self.max_messages:
            return
        system = self.messages[0]       # ALWAYS keep system prompt
        rest = self.messages[1:]
        rest = rest[-(self.max_messages - 1):]  # Keep most recent
        self.messages = [system] + rest

# Long-term: JSON on disk
class LongTermMemory:
    def add_lesson(self, lesson: str):
        if lesson not in self.lessons:  # Dedupe
            self.lessons.append(lesson)
            self._save()  # Persist to disk immediately

    def get_relevant_episodes(self, task: str):
        # Simple keyword overlap (production would use embeddings)
        task_words = set(task.lower().split())
        scored = []
        for ep in self.episodes:
            ep_words = set(ep["task"].lower().split())
            overlap = len(task_words & ep_words)
            scored.append((overlap, ep))
        scored.sort(key=lambda x: -x[0])
        return [ep for _, ep in scored[:limit]]
```

**Lines to highlight:**
- `self._trim()` — this simulates context window limits. Real LLMs have token limits (e.g. 128K tokens). Old messages get dropped. The system prompt is ALWAYS kept.
- `self._save()` — persistence. This is what makes the agent remember across sessions.
- `get_relevant_episodes()` — retrieval. In production, you'd use vector embeddings (like Pinecone, ChromaDB). Here we use keyword overlap — simpler, honest, teachable.

### TEACHING TIP
Show students the `memory/long_term.json` file after running the demo. They can open it in a text editor and SEE the episodes and lessons. Make it tangible.

### DISCUSSION QUESTIONS
1. What's the difference between this and RAG? (RAG retrieves documents from external sources. This retrieves the agent's OWN past experiences. Same concept, different data source.)
2. Why not keep everything in short-term memory? (Token limits = cost + latency. You can't send 1M tokens every API call.)
3. How would you improve `get_relevant_episodes()`? (Use embeddings for semantic search instead of keyword matching.)
4. What if the agent "forgets" an important lesson? (In production, you'd want a "consolidation" process that reviews and prunes lessons.)

---

## LAYER 4: TOOLS (`core/tools.py` + `tools/builtin.py`)

### THE CONCEPT
Tools are how agents interact with the world. Without tools, an LLM is just a text generator. With tools, it's an agent that can:
- Search the web
- Run code
- Query databases
- Send emails
- Control APIs
- Anything you can write a function for

This is exactly how OpenAI function calling works, and how MCP (Model Context Protocol) standardizes it across providers.

### THE PROBLEM
The LLM can't DO anything by itself. It can only generate text. Tools bridge the gap between "thinking" and "doing."

### THE ANALOGY
> The LLM is a brain. Tools are hands.
>
> A brain alone can think about building a bookshelf. But it can't actually build one.
> With hands (tools), the brain can: measure wood, cut it, drill holes, assemble pieces.
>
> The more tools you give the brain, the more it can accomplish.
> The BETTER the tools, the better the results.

### THE CODE WALKTHROUGH

```python
# A Tool is just a function + a schema describing it to the LLM
@dataclass
class Tool:
    name: str              # "calculator"
    description: str       # "Evaluate a math expression"
    func: Callable         # The actual Python function
    parameters: dict       # JSON schema for the LLM

    def to_schema(self) -> dict:
        # This is the OpenAI function-calling format
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def call(self, **kwargs) -> str:
        try:
            result = self.func(**kwargs)
            return str(result)
        except Exception as e:
            # Fail soft — return error as string, don't crash
            return f"TOOL_ERROR[{self.name}]: {e}"
```

**Lines to highlight:**
- `to_schema()` — this is what gets sent to the LLM. The LLM reads this schema and decides: "I should call `calculator` with `expression='2+2'`"
- `call()` — the try/except is CRITICAL. In production, tools fail. The agent needs to handle failures gracefully, not crash.
- `parameters` — this is JSON Schema. Same format used by OpenAI, same concept as MCP tool definitions.

### TEACHING TIP
Have students write their OWN tool. Give them 5 minutes to write a tool that:
- Takes a URL
- Fetches the page
- Returns the title

Then register it and run the agent. They'll see the agent call THEIR tool. This is the "aha" moment.

### DISCUSSION QUESTIONS
1. What happens if a tool's description is wrong? (The LLM will misuse it. Tool descriptions are PROMPT ENGINEERING.)
2. Should tools validate their inputs? (Yes — but the LLM might pass bad args anyway. Handle it in `call()`.)
3. How is this related to MCP? (MCP standardizes tool definitions across providers. This is a simplified version of the same concept.)
4. Could an agent create a new tool at runtime? (Yes — this is "tool creation" or "meta-tooling." Advanced research area.)

---

## LAYER 5: VISUAL TRACE (`core/trace.py`)

### THE CONCEPT
Agents are black boxes. You send a task, get an answer, but you don't know WHAT HAPPENED INSIDE. This makes them:
- Hard to debug
- Hard to trust
- Hard to learn from

The trace system logs EVERY step — every thought, every tool call, every observation — with timestamps, in real-time, with color-coded output.

### THE PROBLEM
When an agent fails, you need to know WHERE it failed. Did it:
- Plan badly?
- Call the wrong tool?
- Misinterpret the tool result?
- Run out of iterations?

Without a trace, you're guessing. With a trace, you can see exactly what happened.

### THE ANALOGY
> A flight recorder (black box) in an airplane. When a plane crashes, investigators don't guess what happened — they read the recorder.
>
> The trace is the agent's flight recorder. Every decision, every action, every result — recorded.

### THE CODE WALKTHROUGH

```python
class TraceStepType(Enum):
    THINK = "🧠 THINK"      # Agent is reasoning
    ACT = "🔧 ACT"         # Agent is calling a tool
    OBSERVE = "👁️ OBSERVE" # Agent is processing results
    REFLECT = "🪞 REFLECT" # Agent is reflecting on failure
    PLAN = "📋 PLAN"       # Agent is breaking down a task
    # ...

class TraceLogger:
    def log(self, step_type, content, data=None):
        step = {
            "timestamp": ...,
            "elapsed_s": ...,  # Time since agent started
            "agent": self.agent_name,
            "type": step_type.value,
            "content": content,
            "data": data,
        }
        self.steps.append(step)    # Store it
        self._render(step)         # Print it with colors
        return step

    def save(self, filename=None):
        # Save full trace to JSON for later analysis
        path.write_text(json.dumps(self.steps, indent=2))
```

**Lines to highlight:**
- `elapsed_s` — timing matters. If an agent takes 30 seconds on one step, something's wrong.
- `self._render(step)` — real-time display. Students can watch the agent think.
- `save()` — the trace is persisted. You can analyze it later, compare runs, etc.

### TEACHING TIP
Run the demo with the trace visible. Pause after each step and ask: "What will the agent do next?" Students predict, then see if they're right. This builds intuition for agent behavior.

### DISCUSSION QUESTIONS
1. Why is observability important in AI systems? (Trust, debugging, compliance, improvement)
2. What would you do if the agent keeps looping on the same step? (Probably a bad system prompt or broken tool. Check the trace.)
3. How does this compare to LangSmith or Langfuse? (Same concept — tracing for LLM apps. This is a simplified version.)

---

## LAYER 6: PLANNER AGENT (`agents/planner.py`)

### THE CONCEPT
Task decomposition. Taking a big, vague task and breaking it into small, specific, actionable subtasks.

This is what project managers do. It's also what the "orchestrator" agent does in systems like AutoGPT, Devin, and SWE-Agent.

### THE PROBLEM
"Design a web application" is too big for one agent to handle in one reasoning loop. But:
1. "Identify the core features needed"
2. "Choose the tech stack"
3. "Design the database schema"
— each of these is tractable.

### THE ANALOGY
> A general contractor doesn't build a house alone. They:
> 1. Break the project into subtasks (foundation, framing, plumbing, electrical, roofing)
> 2. Assign each to a specialist
> 3. Coordinate the order (foundation before framing, framing before roofing)
>
> The Planner is the general contractor.

### TEACHING TIP
Show students what happens with a bad plan vs a good plan:
- Bad plan: "Do the whole thing" → Worker struggles, produces vague output
- Good plan: 3 specific subtasks → Workers produce detailed, useful output

### DISCUSSION QUESTIONS
1. What makes a good decomposition? (Each subtask should be self-contained, specific, and ordered)
2. Could the Planner use tools? (Yes — it could search for similar projects, read requirements docs)
3. What if the Planner's decomposition is wrong? (The whole pipeline fails. The Planner is the most critical agent.)

---

## LAYER 7: WORKER AGENT (`agents/worker.py`)

### THE CONCEPT
The "hands" of the system. Takes a subtask, uses the ReAct loop, produces a concrete output.

Workers are interchangeable and parallelizable. 5 subtasks → 5 workers running simultaneously.

### THE ANALOGY
> Construction workers on a house. The contractor (Planner) says "build the foundation." The worker (Worker) does it — measuring, mixing concrete, pouring. They use tools (trowel, level, mixer). They make decisions on the fly.

### TEACHING TIP
Show that Workers inherit from BaseAgent — they get the full ReAct loop. The only difference is the system prompt (which tells them to focus on ONE subtask).

### DISCUSSION QUESTIONS
1. Should all workers use the same model? (No — you might use a cheaper model for easy subtasks, a smarter model for hard ones.)
2. Can workers talk to each other? (Not in this system. In production, you might want inter-agent communication.)
3. What if two workers produce contradictory outputs? (The Judge resolves this.)

---

## LAYER 8: CRITIC AGENT (`agents/critic.py`)

### THE CONCEPT
Adversarial reasoning. A separate agent whose ONLY job is to find flaws in other agents' work.

From research on "Multi-Agent Debate" (Du et al., 2023): having agents argue produces better results than a single agent reasoning alone.

### THE PROBLEM
When an agent produces an answer, it's confident. But it might be wrong. If the same agent evaluates its own work, it has a self-validation bias (it tends to approve its own output).

A SEPARATE agent with a different prompt (adversarial) catches errors the original agent misses.

### THE ANALOGY
> In academia, peer review works because the reviewer is DIFFERENT from the author. If you review your own paper, you'll miss your own blind spots.
>
> The Critic is the peer reviewer. It didn't write the answer, so it has no ego invested in it being right.

### TEACHING TIP
Show the Critic's output to students. Point out that it finds REAL problems — vague claims, missing evidence, logical gaps. This is not rubber-stamping.

Then show what happens when the Critic is too strict (rejects everything, agent loops forever) vs too lenient (approves everything, quality drops). Discuss the tradeoff.

### DISCUSSION QUESTIONS
1. Could the same LLM be both Worker and Critic? (Yes, but it works better when they have different prompts — less self-validation bias.)
2. Is the Critic always right? (No — it can be wrong too. That's why the Judge does final synthesis, not the Critic.)
3. How is this different from a human reviewer? (Faster, cheaper, consistent — but less nuanced. It's a tool, not a replacement.)

---

## LAYER 9: JUDGE AGENT (`agents/judge.py`)

### THE CONCEPT
"Agent-as-a-Judge" — using an agent to evaluate and synthesize the output of other agents.

The Judge doesn't just concatenate outputs. It:
1. Reads all worker outputs
2. Identifies conflicts between them
3. Resolves conflicts
4. Synthesizes into a coherent answer
5. Scores the quality

### THE PROBLEM
When 3 workers produce 3 pieces of work, you need to stitch them together. But they might:
- Contradict each other
- Overlap
- Leave gaps
The Judge handles this.

### THE ANALOGY
> An editor at a newspaper. Three journalists each cover a piece of a story. The editor reads all three, spots where they disagree, picks the best version, and writes a single coherent article.

### DISCUSSION QUESTIONS
1. Could you replace the Judge with simple concatenation? (You'd get a disjointed, repetitive answer. The Judge adds coherence.)
2. Should the Judge score quality? Why? (Yes — for feedback loops. Low scores trigger more reflexion rounds.)
3. Could the Judge request more work from workers? (In production, yes — this is an "iterative refinement" loop.)

---

## LAYER 10: ORCHESTRATOR (`agents/orchestrator.py`)

### THE CONCEPT
Meta-agentic behavior — an agent (or system) that uses OTHER AGENTS as its tools.

The Orchestrator doesn't call the LLM directly to solve problems. Instead, it:
1. Calls the Planner to decompose
2. Spins up Workers to execute
3. Calls the Critic to evaluate
4. Triggers Reflexion if needed
5. Calls the Judge to synthesize

This is how production systems work: AutoGPT, Devin, SWE-Agent, and others all have an orchestration layer.

### THE ANALOGY
> A symphony conductor. The conductor doesn't play any instrument. They coordinate the musicians — telling each section when to play, how loud, how fast.
>
> The Orchestrator is the conductor. The agents are the musicians.

### THE CODE WALKTHROUGH

```python
def run(self, task: str) -> dict:
    # 1. PLAN
    subtasks = self.planner.plan(task)

    # 2. EXECUTE EACH SUBTASK
    for subtask in subtasks:
        worker = WorkerAgent(name=f"Worker-{subtask_id}", ...)

        # Wrap with Reflexion + Critic
        reflexion_engine = ReflexionEngine(agent=worker, ...)

        # The Critic is the feedback function for Reflexion
        result = reflexion_engine.run_with_reflexion(
            task=subtask_desc,
            feedback_fn=critic.evaluate,  # ← Critic decides if it's good enough
        )

    # 3. SYNTHESIZE
    final_answer = self.judge.synthesize(task, worker_outputs)
```

**Lines to highlight:**
- `feedback_fn=critic.evaluate` — this is where Critic and Reflexion connect. The Critic evaluates → if rejected → Reflexion generates a lesson → Worker retries.
- The Orchestrator is not an agent itself — it's a coordinator. In more advanced systems, the Orchestrator could BE an agent that decides dynamically which agents to call.

### TEACHING TIP
Draw the architecture diagram on a whiteboard FIRST. Then show the code. Students should be able to map each box in the diagram to a line of code.

### DISCUSSION QUESTIONS
1. Could the Orchestrator be an LLM-based agent that dynamically decides which agents to call? (Yes — this is how "router" or "dispatcher" agents work in production.)
2. What if one subtask depends on another's output? (In this system, subtasks are independent. In production, you'd need a dependency graph.)
3. How would you parallelize this? (Run workers concurrently with asyncio or threading.)

---

## PUTTING IT ALL TOGETHER: THE FULL FLOW

```
User gives task
      ↓
ORCHESTRATOR receives task
      ↓
PLANNER decomposes into subtasks
      ↓
For each subtask:
    WORKER attempts it (using ReAct loop + tools)
         ↓
    CRITIC evaluates the output
         ↓
    If REJECTED:
        REFLEXION generates a verbal lesson
        Lesson stored in LONG-TERM MEMORY
        WORKER retries with lesson in context
        CRITIC re-evaluates
        (repeat up to max_reflections times)
    If APPROVED:
        Output goes to JUDGE
      ↓
JUDGE synthesizes all outputs → FINAL ANSWER
      ↓
Result returned to user
All traces saved to disk for analysis
```

---

## SUGGESTED TEACHING SEQUENCE

### Session 1: The ReAct Loop (core/agent.py)
- Just `BaseAgent` with one tool
- Show the think-act-observe loop
- Have students trace through a single run

### Session 2: Tools (core/tools.py + tools/builtin.py)
- Students write their own tools
- Register them, run the agent
- See how tool selection works

### Session 3: Memory (core/memory.py)
- Show short-term vs long-term
- Run the agent twice, show it remembering
- Discuss context window limits

### Session 4: Reflexion (core/reflexion.py)
- Show an agent failing without reflexion
- Add reflexion, show it improving
- Read the generated reflections together

### Session 5: Multi-Agent Orchestration
- Add Planner, Critic, Judge
- Run the full pipeline
- Watch the trace output together

### Session 6: Advanced Discussion
- MCP integration
- Production considerations
- What would you change to deploy this at scale?

---

## COMMON STUDENT QUESTIONS (with answers)

**Q: How is this different from LangChain?**
A: LangChain is a production framework with thousands of lines. This is ~500 lines. You can read every line and understand it. LangChain abstracts the loop away; we expose it.

**Q: Can I use this in production?**
A: No — it's a teaching tool. It lacks error handling, rate limiting, async execution, vector memory, and many other production concerns. But the PATTERNS are the same.

**Q: Why OpenAI and not open-source models?**
A: OpenAI's function calling API is the cleanest for teaching. You can swap in any LLM that supports tool calling (Anthropic, open-source via vLLM, etc.) — the architecture doesn't change.

**Q: How does this compare to MCP?**
A: MCP standardizes the tool layer (Layer 4). Our `ToolRegistry` is a simplified MCP. If you replaced `ToolRegistry` with an MCP client, the system would work the same way but use the standard protocol.

**Q: Is Reflexion real learning?**
A: It's in-context learning, not weight updates. The model doesn't change. But the behavior improves within the session. Whether that's "real" learning is a philosophical question worth discussing in class.