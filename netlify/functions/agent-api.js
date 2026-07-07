/**
 * Netlify Serverless Function: agent-api
 * 
 * Runs the multi-agent system and returns results.
 * 
 * WHY JAVASCRIPT HERE:
 * Netlify CLI locally deploys JS functions natively but NOT Python functions.
 * Python functions only work via Git-connected CI/CD. To make this work
 * with `netlify deploy --prod` (local CLI), we use a JS function that
 * calls the OpenAI API directly — implementing the same multi-agent logic
 * as the Python backend.
 * 
 * The Python version (netlify/functions/agent_api.py) is available for
 * Git-connected deployments. Both implementations are equivalent.
 */

// Netlify Node 18+ has native fetch — no node-fetch needed

// --- AGENT PROMPTS (same as Python version) ---
const PLANNER_PROMPT = `You are a PLANNER agent in a multi-agent system.
Your job: Take a complex task and break it into 2-5 subtasks that other agents can work on independently.
Rules:
1. Each subtask must be self-contained
2. Order matters — list them in the order they should be done
3. Be specific
4. If the task is simple enough for one agent, return just one subtask
Output format (STRICT JSON):
{"subtasks": [{"id": 1, "task": "...", "description": "..."}]}
Output ONLY the JSON.`;

const WORKER_PROMPT = `You are a WORKER agent in a multi-agent system.
Your job: Complete the specific subtask assigned to you.
Rules:
1. Be thorough
2. Be concrete and specific
3. Your output will be reviewed by a CRITIC agent
4. Keep focused on your specific subtask
Produce a clear, well-structured answer.`;

const CRITIC_PROMPT = `You are a CRITIC agent in a multi-agent system.
Your job: Find flaws in the work produced by other agents.
Be ruthless but fair.
For each piece of work, evaluate: Correctness, Completeness, Specificity, Assumptions, Counter-arguments.
Output format:
- If acceptable: "APPROVED: <brief note>"
- If needs improvement: "REJECTED: <specific list of problems>"
Be specific.`;

const JUDGE_PROMPT = `You are the JUDGE agent in a multi-agent system.
Your job: Take the outputs of multiple worker agents and synthesize them into ONE final, coherent answer.
Rules:
1. Do NOT just concatenate. SYNTHESIZE.
2. If workers contradict each other, pick the best version and note the conflict.
3. Structure the final answer clearly.
4. At the end, give a quality score from 1-10 with a brief justification.`;

const REFLECTION_PROMPT = (task, attempt, feedback) => `You are a reflective reasoning engine. An agent tried to complete a task but fell short.

## Task:
${task}

## Agent's Attempt:
${attempt}

## Feedback (why it wasn't good enough):
${feedback}

## Your Job:
Generate a CONCISE reflection (2-4 sentences) answering:
1. What went wrong?
2. What should the agent do differently next time?
3. What specific strategy change would help?

Be specific. Output ONLY the reflection, no preamble.`;

// --- HELPER: Call OpenAI ---
async function callLLM(apiKey, model, systemPrompt, userMessage) {
  const response = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model: model,
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userMessage },
      ],
      temperature: 0.7,
    }),
  });

  if (!response.ok) {
    const err = await response.text();
    throw new Error(`OpenAI API error ${response.status}: ${err}`);
  }

  const data = await response.json();
  return data.choices[0].message.content;
}

// --- HELPER: Parse plan JSON ---
function parsePlan(raw) {
  try {
    let jsonStr = raw;
    if (jsonStr.includes('```json')) {
      jsonStr = jsonStr.split('```json')[1].split('```')[0];
    } else if (jsonStr.includes('```')) {
      jsonStr = jsonStr.split('```')[1].split('```')[0];
    }
    const data = JSON.parse(jsonStr.trim());
    return data.subtasks || [{ id: 1, task: raw, description: raw }];
  } catch (e) {
    return [{ id: 1, task: raw, description: raw }];
  }
}

// --- MAIN HANDLER ---
exports.handler = async (event, context) => {
  try {
    if (event.httpMethod !== 'POST') {
      return {
        statusCode: 405,
        body: JSON.stringify({ error: 'Method not allowed. Use POST.' }),
      };
    }

    const body = JSON.parse(event.body || '{}');
    const task = (body.task || '').trim();
    const model = body.model || 'gpt-4o';
    const maxReflections = body.max_reflections !== undefined ? body.max_reflections : 2;
    const apiKey = body.api_key || '';

    if (!task) {
      return { statusCode: 400, body: JSON.stringify({ error: 'No task provided' }) };
    }
    if (!apiKey) {
      return { statusCode: 400, body: JSON.stringify({ error: 'No API key provided' }) };
    }

    const traceEvents = [];
    const startTime = Date.now();

    function logTrace(type, agent, content, data = {}) {
      traceEvents.push({
        type: 'trace',
        step_type: type,
        agent,
        content,
        data,
        elapsed: ((Date.now() - startTime) / 1000).toFixed(3),
      });
    }

    // --- STEP 1: PLAN ---
    logTrace('plan', 'Planner', `Decomposing: ${task}`);
    logTrace('think', 'Planner', 'Asking Planner to decompose task...');
    const planRaw = await callLLM(apiKey, model, PLANNER_PROMPT, task);
    const subtasks = parsePlan(planRaw);
    logTrace('success', 'Planner', `Decomposed into ${subtasks.length} subtasks`, { subtasks: subtasks.map(s => s.task) });

    // --- STEP 2: EXECUTE EACH SUBTASK ---
    const workerOutputs = [];

    for (let i = 0; i < subtasks.length; i++) {
      const subtask = subtasks[i];
      const subtaskDesc = subtask.task || subtask.description || JSON.stringify(subtask);
      const workerName = `Worker-${subtask.id || i + 1}`;

      logTrace('plan', 'Orchestrator', `Processing subtask ${i + 1}: ${subtaskDesc.substring(0, 80)}...`);

      let currentTask = subtaskDesc;
      let answer = '';
      let succeeded = false;
      const attempts = [];

      for (let round = 0; round <= maxReflections; round++) {
        logTrace('think', workerName, `Attempt ${round + 1} for: ${subtaskDesc.substring(0, 60)}...`);

        // Run worker
        answer = await callLLM(apiKey, model, WORKER_PROMPT, currentTask);
        logTrace('observe', workerName, `Produced ${answer.length} chars`);

        // Evaluate with critic
        logTrace('decide', 'Critic', `Evaluating work for: ${subtaskDesc.substring(0, 60)}...`);
        const criticInput = `## Original Task:\n${subtaskDesc}\n\n## Work to Evaluate:\n${answer}\n\nNow evaluate it.`;
        const criticResult = await callLLM(apiKey, model, CRITIC_PROMPT, criticInput);
        const isApproved = criticResult.trim().toUpperCase().startsWith('APPROVED');

        if (isApproved) {
          logTrace('success', 'Critic', `Attempt ${round + 1} approved!`);
          attempts.push({ attempt: answer, reflection: null, success: true });
          succeeded = true;
          break;
        }

        logTrace('error', 'Critic', `Rejected: ${criticResult.substring(0, 200)}`);

        // Generate reflection
        logTrace('reflect', `Reflexion-${workerName}`, `Generating reflection for round ${round + 1}...`);
        const reflection = await callLLM(apiKey, model, 'You are a reflection engine.', 
          REFLECTION_PROMPT(subtaskDesc, answer, criticResult));
        logTrace('reflexion', `Reflexion-${workerName}`, `Reflection: ${reflection.substring(0, 100)}...`);

        attempts.push({ attempt: answer, reflection, success: false });

        // Inject lesson for next attempt
        currentTask = `${subtaskDesc}\n\n## LESSON FROM PREVIOUS ATTEMPT:\n${reflection}\n\nPlease try again, applying this lesson. Do NOT repeat the same mistakes.`;
      }

      workerOutputs.push({
        subtask: subtaskDesc,
        output: answer,
        attempts,
        approved: succeeded,
        total_reflections: succeeded ? attempts.length - 1 : attempts.length,
      });
    }

    // --- STEP 3: SYNTHESIZE ---
    logTrace('think', 'Judge', `Synthesizing ${workerOutputs.length} worker outputs...`);
    let outputsText = '';
    for (const w of workerOutputs) {
      outputsText += `\n### Subtask: ${w.subtask}\n${w.output}\n`;
    }
    const judgeInput = `## Original Task:\n${task}\n\n## Worker Outputs (by subtask):\n${outputsText}\n\nSynthesize these into a final answer. Remember: unify, don't concatenate.`;
    const finalAnswer = await callLLM(apiKey, model, JUDGE_PROMPT, judgeInput);
    logTrace('success', 'Judge', `Synthesis complete (${finalAnswer.length} chars)`);

    const result = {
      task,
      subtasks,
      worker_outputs: workerOutputs,
      final_answer: finalAnswer,
      total_time_s: ((Date.now() - startTime) / 1000).toFixed(2),
    };

    // Build SSE response
    let sseBody = '';
    for (const evt of traceEvents) {
      sseBody += `data: ${JSON.stringify(evt)}\n`;
    }
    sseBody += `data: ${JSON.stringify({ type: 'result', data: result })}\n`;

    return {
      statusCode: 200,
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
      },
      body: sseBody,
    };

  } catch (error) {
    console.error('Function error:', error);
    return {
      statusCode: 500,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ type: 'error', message: `Internal error: ${error.message}` }),
    };
  }
};