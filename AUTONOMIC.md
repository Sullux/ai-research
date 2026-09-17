# Autonomic Finite State Machine (A-FSM)

## 1. Executive Summary & Biological Analogy

Contemporary Large Language Model (LLM) agent frameworks operate under a flawed assumption: that every decision, transition, and evaluation must pass through the model's deliberative, conscious reasoning channel. In standard agentic loops (e.g., ReAct, LangChain, AutoGPT), an LLM is prompted with a monolithic textual context and asked to produce structured JSON or code to decide its next step. This design introduces multi-second round-trip latencies, inflates context windows with ephemeral clutter, introduces severe non-determinism, and induces reasoning fatigue.

In biological nervous systems, the brain does not operate as a single monolithic deliberative loop. Instead, it is partitioned into two distinct systems:
1. **The Autonomic Nervous System (Brainstem & Reflex Arcs):** Subconscious, millisecond-level feedback loops that regulate pupil dilation, cardiac rhythm, swallowing reflexes, and motor coordination. The autonomic system acts on sensory input rapidly and deterministically without conscious intervention.
2. **The Cerebral Cortex (Conscious Deliberation):** Slow, high-energy, deliberative thought used for novel problem-solving, strategic planning, and reflective reasoning.

The **Autonomic Finite State Machine (A-FSM)** equips the inference engine and host orchestrator with an instinctual nervous system. By executing transient, constrained logit probes directly on the model's latent activations—and immediately rolling back the KV clock—the engine evaluates transitions, routes events, gates thinking channels, and steers progressive stream ingestion in sub-2ms intervals without polluting the KV cache. The deliberative cortex is engaged only when genuine reasoning is required.

---

## 2. Division of Labor: Brainstem Engine vs. Host Orchestrator

To maintain clean separation of concerns and avoid bloated software design:

```
┌─────────────────────────────────────────────────────────────┐
│                       HOST (Node.js)                        │
│                                                             │
│  ┌─────────────────────────┐   ┌─────────────────────────┐  │
│  │    A-FSM Interpreter    │   │   Persistent Context    │  │
│  │  (State Graph & Guards) │◄──┤ (Tasks, Notes, Cursors) │  │
│  └────────────┬────────────┘   └─────────────────────────┘  │
│               │                                             │
│               │ client.probe(prompt, candDigits, 1)         │
│               ▼                                             │
├───────────────┼─────────────────────────────────────────────┤
│  Unix Socket  │ Binary Wire Protocol (OP_PROBE_AUTONOMIC)   │
├───────────────┼─────────────────────────────────────────────┤
│               ▼                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │               BRAINSTEM ENGINE (Zig / GPU)            │  │
│  │                                                       │  │
│  │  • Physical KV Ring Buffer & RoPE (4096 slots)        │  │
│  │  • Fast Constrained Logit Softmax & Entropy           │  │
│  │  • O(1) rollbackClock() Non-Destructive Invariant     │  │
│  │  • Outer TemplateState Turn Conformance               │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

1. **The Brainstem Engine (Zig / Vulkan GPU Compute):**
   * High-performance tensor execution, physical ring buffer geometry, UMA memory synchronization, and canonical chat template framing.
   * Exposes a single, universal reflex primitive over the wire: `OP_PROBE_AUTONOMIC` (`client.probe()`).
   * Evaluates 1-token constrained probes in ~1.5 ms and micro-decodes in ~200–400 ms.
   * Strictly agnostic to application state machines, YAML parsing, and task tree topologies.
2. **The Host Orchestrator (Node.js / TUI):**
   * Evaluates the A-FSM state graph, manages persistent context dictionaries, executes reducers/ops, and binds I/O streams.
   * Compiles YAML state machines on-the-fly (< 5 ms).
   * Drives progressive reading and soft-yield check-in loops.

---

## 3. Disambiguation: Operations (`op`) vs. Tools (`tool`)

Overloading the term "tool" causes severe cognitive confusion across documentation, prompt design, and code. The A-FSM establishes a strict architectural boundary:

* **Tools (`<|tool_call>`)**: Heavyweight, generative operations. The LLM must enter an explicit tool channel, generate structured JSON arguments, and wait for execution. Tools take hundreds of tokens and hundreds of milliseconds of decode time.
* **Operations (`op`)**: Lightweight state transitions and micro-actions executed deterministically by the A-FSM host (`yield`, `reflect`, `resume`, `ask_user`, `emit`, `note`, `recall`, `edit_cell`). They execute in sub-millisecond time without generative overhead.

---

## 4. Mathematical Foundations of `probeAutonomic`

All autonomic decisions are grounded in transient logit evaluation directly from the model's KV attention state.

```
+-------------------------------------------------------------+
|                     probeAutonomic                          |
+-------------------------------------------------------------+
| 1. Record snapshot clock: saved_clock = engine.clock        |
| 2. Wrap prompt in canonical template (formatProbeFrame)     |
| 3. Prefill probe prompt (transient slots)                   |
| 4. If discrete (max_decode_tokens == 1):                    |
|    a. Mask logits to candidate token set                    |
|    b. Compute candidate softmax probabilities and entropy   |
|    c. Select winning candidate via argmax / sample          |
| 5. If generative (max_decode_tokens > 1):                   |
|    a. Autoregressively decode up to max_decode_tokens       |
|    b. Extract decoded text slice                            |
| 6. Rollback KV ring buffer: rollbackClock(saved_clock)      |
| 7. Return winning index, token, confidence, entropy, cost   |
+-------------------------------------------------------------+
```

### 4.1 Normalized Probability (Confidence) & Shannon Entropy
Given unnormalized logits $z_{c_i}$ for each candidate token $c_i \in C$ at temperature $T$:

$$P(c_i \mid C) = \frac{\exp(z_{c_i} / T)}{\sum_{j=1}^K \exp(z_{c_j} / T)}, \quad \text{confidence} = \max_i P(c_i \mid C)$$

$$H(C) = - \sum_{i=1}^K P(c_i \mid C) \ln P(c_i \mid C)$$

* **Low Entropy ($H \to 0$):** Clear consensus; the model has an unambiguous latent bias toward one transition.
* **High Entropy ($H \to \ln K$):** Ambivalence; triggers fallback cascades or conscious deliberation.

---

## 5. Declarative A-FSM Specification (YAML)

The A-FSM markup format is declarative and human-readable. It defines state topology, prompts, candidate action sets, and target operations.

### 5.1 Pure JavaScript State Machine Factories over YAML DSLs
To prevent building an accidental, fragile DSL in YAML (Greenspun's Tenth Rule) or splitting logic between markup and callback registries, the A-FSM is authored directly in **pure, functional JavaScript**:
* **Direct Mathematical Expressiveness:** Evaluating confidence thresholds, entropy ceilings, or cost-adjusted probabilities (e.g. `res.winningIdx === 0 && res.confidence >= 0.80`) uses native JS operators and math without markup AST compilation.
* **Native Template Literals:** Prompts use standard ES6 template strings with full syntax highlighting, refactoring support, and linting.
* **Full Debuggability:** Standard debugging, breakpoints, and logging operate directly inside state transitions.
* **The Context is a plain dictionary:** (`Record<string, any>`) managed functionally.

### 5.2 Schema & Factory Pattern

State machines are exported as functional factories located under `tui/lib/afsm/machines/`:

```javascript
const chatMachineFactory = () => ({
  name: 'chat',
  initial: 'responding',
  states: {
    responding: async (ctx, { probe }) => {
      const prompt = [
        "Regarding the user's message:",
        '0: I have an immediate answer',
        '1: I need to think',
        'Decision:',
      ].join('\n')

      const res = await probe(prompt, ['0', '1'], 1)
      const isDirect = res.winningIdx === 0 && res.confidence >= 0.80

      return {
        op: isDirect ? 'direct_response' : 'think',
        target: 'idle',
        meta: {
          winningIdx: res.winningIdx,
          confidence: res.confidence,
          cost: res.costMs,
          entropy: res.entropy,
          isDirect,
        },
      }
    },
    idle: async () => ({
      op: 'yield',
      target: 'idle',
    }),
  },
})
```

---

## 6. Progressive Reading & Soft-Yield Check-In Loop

One of the primary applications of the A-FSM is **Progressive Stream Ingestion** (reading large files, logs, or multi-turn conversational histories).

### 6.1 The Progressive Ingestion Architecture
Rather than slicing texts by arbitrary regex heuristics or dumping 10 GB into memory, the host pushes raw 512-character blocks directly into the inference engine KV cache. 

At natural syntactic resting points (sentences, code blocks, paragraph ends), the engine triggers an elastic soft yield (`STOP_ELASTIC_YIELD`). The A-FSM immediately executes a 1-token reflex probe directly on the ambient attention state:

```javascript
// tui/lib/afsm/machines/reader.js
const readerMachineFactory = () => ({
  name: 'reader',
  initial: 'reading',
  states: {
    reading: async (ctx, { probe }) => {
      const prompt = [
        'Reading Check-In for active channel:',
        '1: Continue reading next segment',
        '2: Pause to make a distilled note of a key finding',
        '3: Pause to ask the user a clarifying question',
        '0: Reading complete, proceed to synthesis and response',
        'Answer with only the index number:',
      ].join('\n')

      const res = await probe(prompt, ['1', '2', '3', '0'], 1)
      const ops = {
        0: { op: 'pull_next_chunk', target: 'reading' },
        1: { op: 'prepare_note', target: 'reflecting' },
        2: { op: 'ask_user', target: 'awaiting_user' },
        3: { op: 'finalize_reading', target: 'synthesizing' },
      }
      return ops[res.winningIdx] || ops[0]
    },
    reflecting: async (ctx, { probe }) => {
      const prompt = 'Distill the key takeaway from the recent reading segment in 1 to 2 sentences:'
      const res = await probe(prompt, [], 32)
      return { op: 'save_note', target: 'reading', note: res.text }
    },
    // ...
  },
})
```

### 6.2 The Note-Taking Hybrid Model
* **In-Band Attention:** Short reflective notes (15–30 tokens) enter the model's KV attention stream as high-salience semantic anchors.
* **Out-of-Band Persistence:** The Host appends the note to `context.notes`. When the 4,096-slot physical ring buffer undergoes Tier 2 FIFO eviction, older raw text chunks are discarded, while the distilled notes survive both in associative memory (`Hippocampus`) and persistent context.

---

## 7. Dynamic Machine Swapping

State machines can be swapped dynamically based on current user intent:

| Machine | Trigger | Primary Operations |
|---|---|---|
| `chat.js` | Default conversational turn | `yield`, `direct_response`, `think` |
| `reader.js` | Reading files or large logs | `pull_next_chunk`, `prepare_note`, `ask_user` |
| `focus.js` | Inter-channel arbitration | `switch_focus`, `bookmark_interrupt` |

---

## 8. Multi-Channel Context Hygiene: Directional Envelopes

To maintain seamless multi-channel multiplexing across chat UI, terminal streams, and external alerts without delimiter hallucination:

```text
<|turn>user
[from: tg/@sullux] Did the build succeed?
<turn|>
<|turn>model
<|channel>thought
Confirming build status to Telegram.
<channel|>[to: tg/@sullux] Build complete. 0 failures.
<turn|>
```

The streaming parser inspects `[to: <target>]` at turn start and routes tokens to the corresponding sink without requiring artificial JSON tool emissions.
