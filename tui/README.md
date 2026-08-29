# TUI Autonomous Orchestration Architecture

This document records the design, engineering principles, state machine mechanics, and cognitive lifecycle of the interactive TUI client and autonomous streaming orchestrator.

---

## 1. Executive Summary & Core Philosophy

Standard autoregressive language models (including the Gemma 4 family) are fundamentally trained on **discrete, synchronous conversational turns**:
1. Host feeds user prompt: `<|turn>user ... <turn|>`.
2. Model reasons and completes: `<|turn>model <|think|> ... <channel|> ... <turn|>`.
3. Model halts on End-Of-Sequence (`EOS`), idling the GPU until the next user prompt.

This paradigm fails in continuous agentic workflows where a model must:
- Page through long documents (10k+ tokens) without blowing context windows.
- Perform multi-step terminal tasks across dozens of clock ticks.
- Accept live mid-flight interruptions without losing working memory or aborting progress.
- Resume suspended tasks cleanly once an interruption is addressed.

Our architecture breaks out of the turn-based cage using a **"Dumb" Task-Agnostic Orchestrator** paired with **Autoregressive Thought Injection (Thought Prefixing)**, **LIFO Task Stacking**, and the **Persistent Episodic Memory Subsystem**.

---

## 2. Architectural Pillars

```
                     ┌─────────────────────────────────────────┐
                     │          USER INPUT / TIMERS            │
                     └────────────────────┬────────────────────┘
                                          │
                                          ▼
                     ┌─────────────────────────────────────────┐
       ┌────────────►│         AUTONOMOUS ORCHESTRATOR         │
       │             │  - Priority LIFO Task Stack             │
       │             │  - Thought Prefix Synthesizer           │
       │             │  - Timer & Interruption Manager         │
       │             └────────────────────┬────────────────────┘
       │                                  │ Injected Thought Frame
       │                                  ▼
       │             ┌─────────────────────────────────────────┐
       │             │           GEMMA 4 12B ENGINE            │
       │             │  - Dynamic Sliding Ring Buffer (1024)   │
       │             │  - Hippocampus Staging & Debounce       │
       │             │  - GPU-Side Top-64 Softcapping Sampler  │
       │             └────────────────────┬────────────────────┘
       │                                  │
       │                 ┌────────────────┴────────────────┐
       │                 ▼                                 ▼
       │        [Tool Call Generated]            [Direct Public Output]
       │        - `plan(brief, steps)`           - Output printed to user
       │        - `done(summary)`                - If stack empty -> IDLE
       │        - `defer(duration)`              - If stack active -> Next Tick
       │        - `terminal_write(...)`
       │                 │
       └─────────────────┴─────────────────────────────────┘
```

### Principle 1: Task-Agnostic "Dumb" Orchestrator
The orchestrator possesses zero domain knowledge about specific jobs (e.g., "reading files", "writing code", "monitoring builds"). It manages only a lightweight stack of atomic plan steps, timer queues, and execution ticks.

### Principle 2: Low Cognitive Load on 12B-Class Models
Small-to-medium parameter models (12B) frequently hallucinate or loop when required to maintain and mutate large, deeply nested JSON state trees. We reduce the model's tool burden to atomic, zero-overhead primitives (`plan`, `done`, `defer`, `ask_user`) while the orchestrator handles hierarchical ID generation, stack nesting, and state transitions.

### Principle 3: Thought Injection (Thought Prefixing)
Rather than appending verbose system instructions on every turn, the orchestrator feeds an open `<|think|>` channel prefix directly into the engine's autoregressive sampler. By setting the model's self-attention attractor into an immediate cognitive groove, the model requires zero ramp-up tokens to re-orient itself.

### Principle 4: Terminal Paging as the Streaming Substrate
Long files and documents are not ingested through bespoke chunking pipelines. The model reads files through an internal terminal session using standard CLI utilities (`cat`, `head`, `less`). The sliding ring buffer retains the active screen, while the `Hippocampus` automatically captures and indexes evicted context into `.episodic.mem`.

---

## 3. The Task Stack & Interruption Lifecycle

The orchestrator maintains an in-memory **LIFO (Last-In, First-Out) Priority Stack**:

```
[Stack Top (Active)]  ---> Step 1140: Respond to user interruption ("What's the status?")
[Suspended Step]      ---> Step 1138.2: Scan filesystem for one-off package installations
[Parent Plan]         ---> Plan 1138: Write automated update script
```

### Lifecycle Rules:
1. **New Plan Creation**: When the model calls `plan(brief, steps)`, the orchestrator converts the plan into a tree of steps, assigns hierarchical IDs (`1138.1`, `1138.2`, `...`), and pushes Step 1 to the stack top.
2. **Step Completion**: When the active step completes, the model calls `done()`. The orchestrator pops the step, advances to the next sibling step, and ticks inference.
3. **Task Suspension & Deferral**: If a step is blocked (e.g. waiting for a long compilation or background job), the model calls `defer(duration="30s", reason="awaiting build")`. The orchestrator moves the step to the `TimerManager` queue and ticks the next pending item or sleeps.
4. **Live User Interruption**:
   - When user input arrives while a task is executing, the orchestrator does not wipe the KV cache or kill the engine.
   - It pauses the active step, pushes a transient user interaction task to the stack top, and prompts the model.
   - Once the user interaction is fulfilled, the orchestrator pops the interruption task and issues a resume tick for the suspended task.
5. **Idle State**: When the stack is empty (or all tasks are sleeping on timers) and the model emits an end-of-turn delimiter (`<turn|>`), the orchestrator enters zero-CPU sleep until the next user prompt or timer expiration.

---

## 4. User Request Routing: The "Answer vs. Plan" Fork

To avoid unnatural latency on simple questions while ensuring complex action requests are never lost, user turns follow a dual-lane strategy with an automatic safety net:

### A. The Injected Decision Thought
When user input arrives, the orchestrator opens the model turn with:
```
<|turn>model
<|think|>
User request received.
Decision:
- If this is a simple query I can answer directly in one turn, respond immediately.
- If this requires actions, multiple steps, or investigation, call `plan` with a brief summary.
```

### B. Execution Lanes:
1. **Direct Answer (Zero-Task Path):**
   - If the model determines it can answer immediately, it emits text in the public channel (`<|channel>response ... <channel|>`) and closes the turn (`<turn|>`).
   - Output is printed directly to the TUI. No tasks are created. If a prior task was suspended underneath, the orchestrator issues a prompt: *"Resume previous task?"* or automatically resumes.
2. **Structured Multi-Step Plan:**
   - If the model calls `plan(brief, steps)`, the orchestrator creates the plan stack and enters the autonomous tick loop.
3. **Auto-Wrap Safety Net (The "Impulsive Model" Fallback):**
   - If a 12B model bypasses `plan` and immediately invokes a working tool (e.g., `terminal_write`, `recall`), the orchestrator catches the event and automatically synthesizes an implicit task entry: `Task: "<Tool Name>: <Command snippet>"`.
   - **Result:** No action or side-effect is ever executed untracked.

---

## 5. Tool Primitives

| Tool | Parameters | Description |
|---|---|---|
| `plan` | `brief: string`, `steps: string[]` | Creates a new structured plan or child sub-plan. Pushes Step 1 to stack. |
| `done` | `summary?: string` | Marks the current active step as complete and advances stack. |
| `defer` | `duration: string`, `reason?: string` | Suspends active step onto timer queue (`"10s"`, `"2m"`). |
| `ask_user` | `prompt: string` | Prompts user for input/permission (e.g., `sudo` password) and suspends task. |
| `terminal_write` | `input: string` | Sends raw keystrokes/commands to the active pty/terminal session. |
| `recall` | `query: string`, `top_k?: number` | Performs explicit associative memory search and latent KV slab rehydration. |

---

## 6. Thought Injection Templates

On every tick, the orchestrator builds a contextual thought prefix tailored to the current stack state:

### Initial Step Tick:
```
<|turn>model
<|think|>
Focus: Plan [ID] - [Plan Brief]
Active Step [ID.Step]: [Step Brief]
Status: In progress.
Next action:
```

### Post-Interruption Resume Tick:
```
<|turn>model
<|think|>
Interruption handled. Resuming Plan [ID] at Step [ID.Step]: [Step Brief].
Previous context remains active in episodic memory.
Next action:
```

### Timer Expiration Tick:
```
<|turn>model
<|think|>
Timer expired for Step [ID.Step] ([Reason]). Checking terminal/state for updates.
Next action:
```

---

## 7. Cognitive Memory Integration

The orchestrator interfaces seamlessly with the engine's three-tier memory hierarchy:

1. **Tier 1 (Static Anchors):** First 32 tokens (system prompt kernel and tool definitions) permanently pinned.
2. **Tier 2 (Sliding Context):** 1,024 active sliding-window tokens. As terminal outputs and step notes scroll off, they are continuously staged in the `Hippocampus`.
3. **Tier 3 (Upper Reasoning Recall):** Top-128 implicit resonant memories injected into layers 16–47 prior to generation.
4. **Episodic Persistence (`.episodic.mem`):** 
   - Every completed step or debounce flush writes a 64-byte aligned episode header, unit-normalized centroid vector, and full-fidelity multi-layer FP16 $(K, V)$ tensor slab.
   - When a suspended plan is resumed after thousands of conversational tokens, calling `recall()` rehydrates the original task context directly into GPU UMA memory in $<0.2\text{ ms}$.

---

## 8. TUI Client & UI Integration

The frontend client leverages `@sullux/tui` with the following architectural features:

- **Native Caret & Multiline Input:** Adopts the native `input` control introduced in `@sullux/tui` (as demonstrated in `../tui/examples/chat/`) for smooth caret navigation, copy/paste, and history navigation.
- **Dual Pane Layout:**
  - **Main Conversation & Terminal View:** Displays live model reasoning (`<|think|>` channel collapsible), streaming responses, and interactive terminal output.
  - **Live Plan & Task Sidebar:** Real-time visual display of the active plan stack, step checkboxes, completed timestamps, and sleeping timer countdowns.
- **Asynchronous Wire Protocol:** Full duplex binary framing (`OP_STREAM_INPUT`, `OP_TOKEN`, `OP_ABORT`, `OP_MEM_QUERY`) over Unix domain sockets with millisecond debounce timers.
