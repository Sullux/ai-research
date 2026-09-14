# Autonomic Finite State Machine (A-FSM)

## 1. Executive Summary & Biological Analogy

Contemporary Large Language Model (LLM) agent frameworks operate under a flawed assumption: that every decision, transition, and evaluation must pass through the model's deliberative, conscious reasoning channel. In standard agentic loops (e.g., ReAct, LangChain, AutoGPT), an LLM is prompted with a monolithic textual context and asked to produce structured JSON or code to decide its next step. This design introduces multi-second round-trip latencies, inflates context windows with ephemeral clutter, introduces severe non-determinism, and induces reasoning fatigue.

In biological nervous systems, the brain does not operate as a single monolithic deliberative loop. Instead, it is partitioned into two distinct systems:
1. **The Autonomic Nervous System (Brainstem & Reflex Arcs):** Subconscious, millisecond-level feedback loops that regulate pupil dilation, cardiac rhythm, swallowing reflexes, and motor coordination. The autonomic system acts on sensory input rapidly and deterministically without conscious intervention.
2. **The Cerebral Cortex (Conscious Deliberation):** Slow, high-energy, deliberative thought used for novel problem-solving, strategic planning, and reflective reasoning.

The **Autonomic Finite State Machine (A-FSM)** equips the inference engine with an instinctual nervous system. By executing transient, constrained logit probes directly on the model's latent activations—and immediately rolling back the KV clock—the engine evaluates transitions, routes events, gates thinking channels, and selects communication targets in sub-2ms intervals without polluting the KV cache. The deliberative cortex is engaged only when genuine reasoning is required.

---

## 2. Core Architecture & Mathematical Foundations

### 2.1 The Standardized `probeAutonomic` Kernel
All autonomic operations are grounded in a unified low-level execution primitive: `probeAutonomic`. Rather than performing a standard token generation cycle that permanently advances the conversational clock and appends tokens to the ring buffer, `probeAutonomic` executes a temporary forward pass, evaluates the output logits against a constrained vocabulary, and restores the exact KV cache state.

```
+-------------------------------------------------------------+
|                     probeAutonomic                          |
+-------------------------------------------------------------+
| 1. Record snapshot clock: saved_clock = engine.clock        |
| 2. Prefill probe prompt (transient slots)                   |
| 3. If discrete (max_decode_tokens == 1):                    |
|    a. Mask logits to candidate token set                    |
|    b. Compute candidate softmax probabilities and entropy   |
|    c. Select winning token via argmax / sample              |
| 4. If generative (max_decode_tokens > 1):                   |
|    a. Autoregressively decode up to max_decode_tokens       |
|    b. Extract decoded text slice                            |
| 5. Rollback KV ring buffer: rollbackClock(saved_clock)      |
| 6. Return winning index, token, confidence, entropy, cost   |
+-------------------------------------------------------------+
```

### 2.2 Mathematical Metrics: Confidence & Entropy
Probing an LLM via natural language to assess its own confidence (*"On a scale of 1-10, how sure are you?"*) fails due to sycophancy, overconfidence, and calibration drift. The A-FSM measures confidence directly from the raw pre-softmax logit distribution over the candidate token set $C = \{c_1, c_2, \dots, c_K\}$.

#### 1. Normalized Candidate Probability (Confidence)
Given unnormalized logits $z_{c_i}$ for each candidate token $c_i \in C$ at temperature $T$:

$$P(c_i \mid C) = \frac{\exp(z_{c_i} / T)}{\sum_{j=1}^K \exp(z_{c_j} / T)}$$

The **confidence** of the winning token $c^*$ is defined as:

$$\text{confidence} = P(c^* \mid C) = \max_{i} P(c_i \mid C)$$

#### 2. Candidate Logit Entropy (Uncertainty)
To measure whether the model is decisive or conflicted between multiple valid branches, we compute the Shannon entropy across the candidate set:

$$H(C) = - \sum_{i=1}^K P(c_i \mid C) \ln P(c_i \mid C)$$

* **Low Entropy ($H \to 0$):** Strong consensus; the model has an unambiguous latent bias toward one transition.
* **High Entropy ($H \to \ln K$):** High uncertainty or ambivalence; triggers fallback cascades or escalates to conscious thinking.

### 2.3 Hardware Invariants
1. **$O(1)$ Transient Slot Eviction:** When `rollbackClock(saved_clock)` is invoked, dynamic ring buffer slots allocated during the probe are marked inactive, and attention mass counters are decremented.
2. **Unified UMA Coherence:** On unified memory architectures (AMD APUs / RDNA 3.5), GPU-resident buffers (`buf_k_cache`, `buf_v_cache`) and CPU host descriptors remain perfectly synchronized without memory reallocations.
3. **Zero Context Contamination:** The active conversation history never observes the prompt or tokens generated during the probe.

---

## 3. The A-FSM Declarative Markup Language (YAML)

The A-FSM specification language is a declarative format for authoring complex autonomic behavior trees, hierarchical states, and reflexive tool pipelines.

### 3.1 Specification Schema & Syntax

```yaml
# An autonomic state definition
<state_name>:
  prompt: <string>                  # Seed clause presented to the model
  [minCertainty: <float 0.0..1.0>]  # Global confidence floor for state options
  [maxTokens: <int>]                # Maximum decode tokens (default: 1)
  options:                          # Candidate transitions or dynamic provider
    - prompt: <string>              # Candidate completion text / label
      [minCertainty: <float>]       # Option-specific confidence floor
      next: <state_name> | <tool>   # Next state name or tool execution directive
    # OR:
    # { context: <provider_name> }  # Dynamic runtime candidate injection
```

### 3.2 Transition Targets
A transition `next` target can be one of two types:
1. **State Transition (`next: <state_name>`):** Recursively transitions to another autonomic node within the state machine.
2. **Action Primitive (`next: { tool: <built_in_action>, ...params }`):** Terminates the autonomic evaluation and signals the inference engine or client orchestrator to execute an action.

Built-in action primitives include:
* `{ tool: yield }`: Pauses inference and yields control back to the event loop.
* `{ tool: think }`: Enters the canonical reasoning channel (`<|channel>thought\n`).
* `{ tool: respond, [channel: <target>] }`: Begins direct content generation on the designated output channel without thinking.
* `{ tool: read_next, [mode: <chunk_mode>] }`: Advances the task's stateful push-stream reader.
* `{ tool: ack, id: <not_id> }`: Acknowledges and clears an alert interrupt.
* `{ tool: snooze, id: <not_id> }`: Defers an interrupt to the queue tail.

### 3.3 Concrete Example: Comprehensive Conversational & Task Controller

```yaml
on_idle:
  prompt: "In this idle state, I should"
  options:
    - prompt: "maintain my current focus and wait"
      minCertainty: 0.6
      next: { tool: yield }
    - prompt: "review pending tasks and pick what to focus on"
      minCertainty: 0.4
      next: decide_focus
    - prompt: "perform background maintenance and memory consolidation"
      minCertainty: 0.75
      next: { tool: memory_consolidate }

on_input:
  prompt: "The incoming input event"
  options:
    - prompt: "is background telemetry or status that does not require changing focus"
      minCertainty: 0.7
      next: { tool: ack }
    - prompt: "is a steering correction or follow-up to the current active task"
      minCertainty: 0.55
      next: focus
    - prompt: "is a completely new, independent user task"
      minCertainty: 0.65
      next: create_new_task

decide_focus:
  prompt: "Next, the channel or task I will focus on is"
  options: { context: active_tasks } # Dynamically populated with tasks 1..N

focus:
  prompt: "To address this task, I need to"
  options:
    - prompt: "directly answer without needing extended reasoning"
      minCertainty: 0.65
      next: { tool: respond }
    - prompt: "think carefully and reason step-by-step"
      minCertainty: 0.4
      next: { tool: think }
    - prompt: "read the next segment of the source document"
      minCertainty: 0.7
      next: { tool: read_next }
    - prompt: "skim the document structure first"
      minCertainty: 0.75
      next: skim_mode
    - prompt: "invoke an external subshell command"
      minCertainty: 0.8
      next: { tool: cmd_prepare }

skim_mode:
  prompt: "I should skim the content by extracting"
  options:
    - prompt: "headings and structural markdown tags"
      next: { tool: skim, strategy: "headings" }
    - prompt: "the opening sentence of each paragraph"
      next: { tool: skim, strategy: "first_sentence" }
    - prompt: "code block signatures and function declarations"
      next: { tool: skim, strategy: "code_signatures" }

create_new_task:
  prompt: "Summarize this user request in a concise 4 to 10 word title:"
  maxTokens: 14
  next: { tool: task_init }
```

---

## 4. Binary Compilation & Autonomic Bytecode (µOps)

To maintain sub-2ms evaluation times, the inference engine cannot parse YAML, allocate strings, or run regexes on the fast path. The declarative YAML is compiled ahead of time (or at session initialization) into a contiguous binary **Autonomic Action Graph (AAG)**.

### 4.1 Bytecode Structure & Node Layout

```zig
pub const AutonomicOpcode = enum(u8) {
    OP_PROBE_DISCRETE = 0x01, // Formats prompt, samples 1 token from pre-tokenized candidates
    OP_PROBE_DECODE   = 0x02, // Prefills prompt, generates up to max_tokens (generative decode)
    OP_DISPATCH_TOOL  = 0x03, // Executes built-in tool / action primitive
    OP_BRANCH_CONF    = 0x04, // Evaluates min_certainty condition and branches
    OP_HALT           = 0x05, // Terminates state machine
};

pub const AutonomicNode = extern struct {
    id: u16,
    opcode: AutonomicOpcode,
    candidate_count: u8,
    min_certainty_f32: f32,
    prompt_str_offset: u32,
    prompt_str_len: u16,
    candidate_token_ids: [8]u32, // Pre-encoded vocabulary IDs!
    transition_node_ids: [8]u16, // Child state indices
    fallback_node_id: u16,       // Executed if max confidence < min_certainty
};
```

### 4.2 Pre-Tokenized Candidate Evaluation
During compilation, every candidate prompt (`"1: read"`, `"2: think"`) is pre-tokenized against the model's vocabulary. When evaluating `OP_PROBE_DISCRETE`:
1. The engine prefills the prompt slice.
2. The logits of `candidate_token_ids[0..candidate_count]` are extracted directly from GPU scratch memory.
3. Candidate softmax probabilities are computed across the small candidate array in under $50\,\mu\text{s}$.
4. If `confidence >= min_certainty_f32`, execution transitions immediately to `transition_node_ids[winning_idx]`.
5. If confidence falls below the threshold, execution jumps directly to `fallback_node_id`.

---

## 5. Multi-Channel Context Hygiene: Directional Envelopes

A recurring failure mode in multi-channel LLM systems is channel confusion. When an LLM interacts across multiple sinks (e.g., desktop chat, Telegram bot, text-to-speech engine, open log files), standard tool-calling approaches fail because:
1. Emitting JSON tool payloads for everyday replies is slow and unnatural for conversational weights.
2. Models insist on producing a final direct conversational response, repeating what they already sent via the tool.
3. Non-canonical synthetic tokens (`<|to_telegram|>`) cause distribution shift, attention degradation, and delimiter hallucination.

### 5.1 Directional Envelopes inside Canonical Turns
The A-FSM solves this by preserving 100% canonical Gemma 4 chat template syntax (`<|turn>user\n...<turn|>` and `<|turn>model\n...<turn|>`) while embedding **Directional Channel Descriptors** at the head of content blocks.

#### Inbound Envelope (Reading / Ingestion):
```text
<|turn>user
[from: tg/@sullux] Hey, did the ReleaseFast GPU build finish?
<turn|>
```
```text
<|turn>user
[from: trm/build stdout] ninja: build complete. 0 failures.
<turn|>
```

#### Outbound Envelope (Writing / Response):
When the A-FSM determines that a response is directed to a specific channel, the server **pre-seeds the channel header** into the turn start:
```text
<|turn>model
<|channel>thought
Charles is asking via Telegram about the build status. The build terminal indicates zero failures. I will confirm to his Telegram channel.
<channel|>[to: tg/@sullux] Yes, the ReleaseFast GPU build succeeded with 0 failures.
<turn|>
```

#### Multi-Sink Responses:
If the model determines that an action requires multi-destination broadcasting, it can emit multiple blocks within a single turn:
```text
<|turn>model
<|channel>thought
I should update the Telegram chat and also announce the completion over the desktop TTS speaker.
<channel|>[to: tg/@sullux] Build complete. Ready for testing.
[to: audio/tts] GPU compilation completed successfully.
<turn|>
```

### 5.2 Architectural Advantages:
1. **Canonical Context Validity:** The KV cache retains clean, natural language. The model's attention mechanism easily learns that `[from: X]` represents sensory input and `[to: X]` represents motor output.
2. **Zero Hallucination:** The server autonomic engine can pre-seed `[to: <selected_channel>]\n` as forced tokens immediately following `<channel|>`. The model never has to guess the channel formatting.
3. **Causal Client Demultiplexing:** The streaming parser monitors for `[to: <target>]` headers. When encountered, it binds the downstream token pipe to that specific target (Telegram HTTP webhook, TTS audio synthesis queue, or chat UI) until a new header or turn boundary arrives.

---

## 6. Autonomic Turn Gating & Reflex Controls

### 6.1 The 1-Token Thinking Gate
Instead of relying on the model to freely decide whether to spend 500 tokens reasoning about a trivial greeting, the engine implements a 1-token reflex gate at the initiation of every turn:

```
[Inbound Prompt]
       │
       ▼
[Autonomic Reflex Probe]
Prompt: "For this user prompt, deliberative reasoning is: [0: Unnecessary, 1: Essential]"
Candidates: ['0', '1']
       │
       ├─────────────────────────────────┐
       ▼                                 ▼
 winning = '0'                     winning = '1'
 (Confidence >= 0.70)              (Or Confidence < 0.70 fallback)
       │                                 │
       ▼                                 ▼
Inject: Direct response            Inject: <|channel>thought\n
Set: suppress_thinking = true      Set: suppress_thinking = false
Result: Instant response (27 t/s)  Result: Full deep reasoning
```

### 6.2 Autonomic Barge-In Arbitration
When an interrupt event arrives during active model generation:
1. Active forward passes pause at the current micro-step.
2. The engine formats a transient 1-token probe:
   ```text
   Active task: <current_task_title>
   Incoming event: <event_preview>
   Directive: Should this event immediately interrupt the active response?
   Decision: [0: Queue until turn end, 1: Interrupt immediately]
   ```
3. If `0`: Inference resumes seamlessly. The event is enqueued into `PENDING` without dropping tokens.
4. If `1`: The engine forces a clean syntactic closure (`... [interrupted] <turn|>`), preserves the partial KV cache, and switches to the interrupt handler.

---

## 7. Roadmap & Implementation Phases

| Phase | Component | Description |
|---|---|---|
| **Phase 1** | **Unified `probeAutonomic` Kernel** | Consolidate `handleTaskTriage`, `handleBacklogTriage`, and `handleTaskTitle` in `src/server.zig` into a reusable, hardened primitive with normalized probability and entropy calculation. |
| **Phase 2** | **1-Token Thinking Gate** | Implement pre-turn reflex evaluation to suppress unnecessary `<|channel>thought` phases on simple conversational turns. |
| **Phase 3** | **Directional Envelopes & Channel Demux** | Standardize `[from: ...]` and `[to: ...]` headers in prompts and implement client-side stream routing to multi-channel sinks. |
| **Phase 4** | **A-FSM Bytecode Compiler & Runtime** | Implement parser for `.afsm.yaml` behavior trees compiling to binary µOps evaluated natively inside `src/server.zig`. |
