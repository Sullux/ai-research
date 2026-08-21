# Memory Architecture: Dual-Process Cognitive Storage & Retrieval

## 1. Overview & Conceptual Foundations

Traditional Large Language Models are fundamentally amnesic between inference calls and brittle within long contexts. Standard Retrieval-Augmented Generation (RAG) and naive sliding context windows suffer from three fatal flaws:
1. **Context Window Exhaustion:** Attention complexity scales poorly, degrading reasoning quality over long sessions.
2. **Textual Impedance Mismatch:** Serializing state into natural language text strings strips away high-dimensional nuance and requires costly re-tokenization and deep prefill re-computation.
3. **Flat, Monolithic Perception:** Standard architectures treat all historical data identically, failing to distinguish immediate sensory perception from subconscious associations and deliberate, focused retrospection.

This architecture models memory on the **Dual-Process Cognitive Architecture** of mammalian neurobiology:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   DUAL-PROCESS MEMORY ARCHITECTURE                     │
├───────────────────────────────────┬────────────────────────────────────┤
│ 1. IMPLICIT PASSIVE RESONANCE     │ 2. EXPLICIT FOREGROUND REPLAY      │
│    (Subconscious Background)      │    (Deliberate Mental Simulation)  │
│    - Tier-3 Peripheral KV Slots   │    - Primary 1,024-Token Stream    │
│    - Autonomous Salience Gating   │    - Full 48-Layer Reasoning       │
│    - Low-bandwidth "tag-along"    │    - Replaces sensory focus        │
└───────────────────────────────────┴────────────────────────────────────┘
```

---

## 2. The Hippocampal Debounce & Consolidation Window

### Biological Analogy: The 6-Second Boundary
In humans, raw sensory transients in echoic/working memory require approximately 2–6 seconds of stable neural rehearsal before hippocampal consolidation commits them into long-term episodic memory. A sudden traumatic interruption (being "knocked out") causes retrograde amnesia for the preceding few seconds because the uncommitted staging window was disrupted before synaptic consolidation could occur.

### Silicon Implementation
Writing every individual token activation ($N=1$) to disk or scanning large vector archives on every decoding cycle introduces intolerable GPU pipeline stalls and bus contention. Furthermore, individual sub-tokens (e.g. `"ing"`, `" the"`) carry virtually zero episodic semantic meaning.

The engine introduces an in-memory **Consolidation Staging Buffer** operating as a temporal debounce:

```
[ Active Decode Stream / Ingestion ]
                 │
                 ▼
┌────────────────────────────────────────────────────────┐
│  Tier-2 Fast GPU Ring Buffer (Sliding Working Window)  │
│  - KV Cache updated per token on GPU                   │
│  - Hidden activation x retained in staging buffer      │
└────────────────────────┬───────────────────────────────┘
                         │
        [ Consolidation Debounce Trigger ]
        - Condition A: Inactivity / Turn boundary (<turn|>)
        - Condition B: Staging buffer capacity (e.g., 32 tokens)
        - Condition C: High activation velocity (||Δx|| > threshold)
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│  Episodic Consolidation Pass (Background / Non-blocking)
│  1. Compute consolidated state diff: Δx = x_t - x_prev │
│  2. Snapshot top-layer semantic vector into DiffArchive│
│  3. Snapshot multi-layer KV states into RAM cache      │
│  4. Async-flush binary Diff record to NVMe store       │
└────────────────────────────────────────────────────────┘
```

---

## 3. Non-Destructive Interruption: Emergence vs. Administrative Abort

### The Cognitive Reality
In standard stateless LLM APIs, an interruption is treated as a destructive network cancel: generation is severed, the partial output is discarded, and the next turn starts anew.

In human cognition, being interrupted mid-thought (e.g., answering a verbal question while writing code) does **not** erase the thought. The half-formed reasoning trajectory is retained in working memory as an open, suspended cognitive thread.

### Two Levels of Interruption

#### 1. Emergent Conversational Interruption (Continuous Streaming Ingestion)
In a true full-duplex streaming architecture, natural conversational interruption is **not an out-of-band control packet**. It is an emergent model behavior driven directly by continuous input:
- As the model generates output, input (microphone audio codec latents, keystrokes, sensor events) streams continuously into Layer 0 via `OP_STREAM_INPUT`.
- **Hierarchical Quiescence as a Noise Sieve:** Ambient acoustic noise (clothing rustle, keyboard clicks, distant chatter) produces near-zero semantic delta ($\|\Delta_{\text{state}}\| < \tau$) at Layer 15. The upper association layers (32–47) actively decoding the response remain undisturbed.
- **Salient Speech Triggers Natural Re-Routing:** When meaningful speech arrives (*"Wait, include an `isMissing` boolean..."*), lower layers emit a significant delta. Upper layers wake up, cross-attend to the new input in the ring buffer, and the output logits naturally pivot in real time (*"Got it, adding `isMissing: bool`..."*).

#### 2. Administrative Emergency Brake (`OP_ABORT`)
When the host application or human user explicitly triggers an immediate stop (e.g. hitting `Ctrl+C` or a UI stop button):
1. The host sends `OP_ABORT` over the binary protocol.
2. Token decoding halts immediately.
3. **The uncommitted staging buffer and active KV slots are NOT discarded.**
4. The active episode is committed to the `DiffArchive` tagged with `is_interrupted = true`:
   ```zig
   pub const MemoryMeta = struct {
       timestamp: u64,
       last_accessed: u64,
       access_count: u32,
       salience_norm: f32,
       layer_id: u8,
       is_interrupted: bool = true,
   };
   ```
5. When subsequent input arrives, the model's attention mechanism still perceives the suspended thought in its sliding window, enabling seamless resumption or retrospection.

---

## 4. Dual Retrieval Modalities: Implicit vs. Explicit

```
┌────────────────────────────────────────────────────────────────────────┐
│                        3-TIER CONTEXT LAYOUT                           │
├───────────────────────┬────────────────────────┬───────────────────────┤
│ Tier 1: Anchors       │ Tier 2: Sliding Window │ Tier 3: Implicit Rec. │
│ [e.g., 32 slots]      │ [e.g., 512 slots]      │ [e.g., 96 slots]      │
│ Core system identity  │ Immediate chronological│ Subconscious resonance│
│ & task directives     │ perceptual stream      │ background priming    │
└───────────────────────┴────────────────────────┴───────────────────────┘
```

### 1. Implicit Memory (Subconscious Resonance)
- **Target:** Tier-3 peripheral slots in the active GPU ring buffer.
- **Trigger:** Autonomous per-step background salience scan:
  $$\text{Salience}(M_i) = \alpha \cdot \cos(x_{\text{current}}, M_i) + \beta \cdot e^{-\lambda \Delta t} + \gamma \cdot \|\Delta_i\|$$
- **Behavior:** Operates quietly in the background without user intervention or conversational interruption. Injects top-matched KV representations into Tier-3 slots, providing subtle associative priming.

### 2. Explicit Memory (Foreground Mental Replay)
- **Target:** The primary 1,024-token streaming ingestion pipeline (Layer 0 $\to$ Layer 47).
- **Trigger:** Host or model-directed `OP_MEM_QUERY` (via tool calling or user command).
- **Behavior:** When the model deliberately stops to "think back", memory retrieval **takes over the primary stream of attention**, exactly like human mental imagery.

#### Advantages of Foreground Streaming for Explicit Recall:
1. **Full-Fidelity Narrative Context:** 512–1,024 tokens allow complete multi-paragraph historical episodes to be examined with zero loss of detail.
2. **Deep 48-Layer Relational Reasoning:** The recalled memory passes through all 48 transformer layers, undergoing full cross-attention with the active query and reasoning goal.
3. **Clean Provenance Delimiters:** Injected memories are wrapped in structural cognitive tags to prevent hallucination:
   ```xml
   <|start_recalled_memory timestamp="12450" mode="fulltext" query="Jane's email"|>
   [Historical episodic content and context]
   <|end_recalled_memory|>
   ```
4. **Natural Sliding Retention:** The recalled episode slides smoothly into the 4,096-token working context buffer, remaining accessible for several subsequent reasoning turns.

---

## 5. Explicit Query Mechanics: `keywords` vs. `fulltext`

The explicit query control plane (`OP_MEM_QUERY`) supports dual search modalities to balance instant retrieval with deep semantic comprehension:

```
                      ┌────────────────────────┐
                      │      OP_MEM_QUERY      │
                      └───────────┬────────────┘
                                  │
         ┌────────────────────────┴────────────────────────┐
         ▼                                                 ▼
┌─────────────────────────────────┐       ┌─────────────────────────────────┐
│     KEYWORD QUERY MODALITY      │       │     FULLTEXT QUERY MODALITY     │
│ - Direct unquantized embedding  │       │ - 1-Pass short forward prefill  │
│   table lookup: E[token_id]     │       │   to obtain Layer 47 vector x   │
│ - Zero GPU compute / <0.1ms     │       │   capturing relations & syntax  │
│ - Exact entity/term matching    │       │ - High-level semantic concept   │
│   ("Jane", "invoice #402")      │       │   matching across complex ideas │
└─────────────────────────────────┘       └─────────────────────────────────┘
```

### Composite Scoring Equation
$$\text{Score}(M_i) = w_{\text{kw}} \cdot \cos(E_{\text{kw}}, M_i) + w_{\text{sem}} \cdot \cos(x^{(47)}_{\text{fulltext}}, M_i) + w_{\text{time}} \cdot \text{Proximity}(t_i, t_{\text{ref}}) + w_{\text{pin}} \cdot \text{IsPinned}(M_i)$$

- **Temporal Walk ("What did I do next?"):** Set $w_{\text{time}} = 1.0$ with $t_{\text{ref}} > t_{\text{event}}$ to step sequentially forward through historical episodes.
- **Continuation Cursor (`cursor_id`):** Allows paginated retrieval ("keep digging") without re-running vector projections.

---

## 6. High-Layer Semantic Search & Multi-Layer KV Rehydration

### High-Layer Semantic Targeting
In our hierarchical model with temporal quiescence, lower layers process lexical and syntactic details, while upper layers (e.g. Layers 32–47 in Gemma 4 12B) process abstract concepts, intent, and causal structure.

- **Storage & Search:** Vector similarity searches are performed against **Layer 47 hidden states** ($x^{(47)}$), yielding noise-free, high-concept semantic matching at $1/48\text{th}$ the compute overhead.

### Full-Stack KV Rehydration
When an episodic memory is selected:
- The `DiffArchive` stores pre-computed Key and Value state snapshots across **all 48 layers simultaneously**.
- When rehydrated, all 48 layers in the active context receive their corresponding high-precision KV representations, ensuring that syntactic, relational, and conceptual attention heads function in perfect harmony.

---

## 7. The Bridge to Phase 4: Synaptic Weight Consolidation ($\Delta W$)

Biological brains physically consolidate memories into synaptic wiring when the energetic cost of repeatedly re-processing an idea in working memory exceeds the metabolic cost of permanent consolidation:

$$\sum \text{Working Memory Tax}(P) > \text{Plasticity Risk Tax}(P)$$

In this architecture:
1. When an episodic memory is explicitly streamed to the foreground repeatedly across multiple sessions, its recurrent activation generates large cumulative activation gradients across the layers.
2. When this cumulative salience surpasses the plasticity threshold, the engine consolidates that episode directly into the layer MLP weight deltas ($\Delta W$) via fast online micro-LoRA / Hebbian updates.
3. **Result:** Once consolidated into the model weights, that knowledge requires **zero context slots** and **zero retrieval latency**.
