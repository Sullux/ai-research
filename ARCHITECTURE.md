# Technical Architecture & Phased Implementation Blueprint

## 1. System & Hardware Target

* **Host Platform:** Linux (Ubuntu x86_64, Framework Desktop).
* **Compute / Memory:** AMD Unified Memory Architecture (UMA) with 96 GB shared RAM allocated to integrated GPU/compute.
* **Storage Subsystem:** PCIe 4.0 NVMe SSD (up to ~7,000 MB/s sequential read/write).
* **Primary Implementation Language:** **Zig** (zero-overhead memory control, explicit allocators, direct C/Vulkan interop, SIMD intrinsics, and native `io_uring` support).
* **Target Model Family (Phase 1):** Gemma 2/4 & LLaMA 3 (GGUF / `.safetensors` model weights).

---

## 2. Universal Data Structures & Binary Layouts

To ensure seamless forward compatibility across all three development phases, the low-level data structures and telemetry headers are defined up front.

### A. The 4,096-Vector Layer Buffer
Each layer $l \in [0, N-1]$ manages a fixed-size contiguous buffer strictly bounded to 4,096 physical token slots ($4096 \times \text{KV\_DIM} \times 4\text{ bytes} \approx 16\text{ MB}$ per layer, $\approx 768\text{ MB}$ total across 48 layers).

The buffer uses a deterministic 3-tier partitioned geometry with a depth-asymmetric Tier 3 allocation:

```
Lower Layers (0 .. 15): [Total Fixed Allocation: 4096 Slots]
┌───────────────────────────────────────┬────────────────────────────────────────────────────────┐
│ Tier 1: Dynamic System Anchors        │ Tier 2: Sliding FIFO Ring Buffer                       │
│ (System Prompt, Persona & Tool Schemas│ (Continuous Streaming Dialogue & Local Context)        │
│ [N_sys ~384 slots - Locked Forever]   │ [4096 - N_sys ~ 3712 slots]                            │
└───────────────────────────────────────┴────────────────────────────────────────────────────────┘

Upper Layers (16 .. 47): [Total Fixed Allocation: 4096 Slots]
┌──────────────────────────────────┬──────────────────────────────────────────┬──────────────────┐
│ Tier 1: Dynamic System Anchors   │ Tier 2: Sliding FIFO Ring Buffer         │ Tier 3: Recall   │
│ (System Prompt & Tool Schemas)   │ (Continuous Streaming Dialogue & Context)│ (Dense Memories) │
│ [N_sys ~384 slots - Locked]      │ [4096 - N_sys - 128 ~ 3584 slots]        │ [128 slots]      │
└──────────────────────────────────┴──────────────────────────────────────────┴──────────────────┘
```

#### Memory Partitioning & Ring Lifecycle Rules:
1. **Tier 1 (Dynamic Anchors):** On session initialization, the exact system prompt length ($N_{\text{sys}} \approx 384$ tokens) is measured and locked permanently into physical slots $[0 \dots N_{\text{sys}}-1]$ across all 48 layers. These slots are marked immutable and never evicted.
2. **Tier 2 (Sliding Ring):** For tokens beyond $N_{\text{sys}}$, the circular write head advances strictly within the Tier 2 window $[N_{\text{sys}} \dots N_{\text{sys}} + W - 1]$ via the modulo formula:
   $$\text{Physical Slot} = N_{\text{sys}} + \left((\text{clock} - N_{\text{sys}}) \pmod{W}\right)$$
   The write head never touches or wraps into the Tier 1 anchor partition $[0 \dots N_{\text{sys}}-1]$.
3. **Tier 3 (Hippocampal Recall):** Allocated exclusively to middle and upper layers (16–47) where factual extraction, entity binding, and reasoning occur. As context rolls off Tier 2, it is compressed into episodic diffs; on associative query match, Top-$K$ dense vectors are injected directly into Tier 3 slots $[4096 - 128 \dots 4095]$.
4. **Zero-Branch Indirect GPU Attention:** The GPU decode kernel gathers active KV entries via an indirection table (`Active_slots`), executing contiguous vector dot products across all active anchors, sliding ring entries, and recall slots in parallel without shader branching.

### B. Dual-Mode Memory Engine: Implicit Priming vs. Explicit Latent Rehydration

The cognitive memory subsystem operates across two complementary modalities:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                DUAL-MODE MEMORY ARCHITECTURE                                │
├─────────────────────────────────────────────────────────────┬───────────────────────────────┤
│ 1. IMPLICIT MEMORY (Subconscious Priming)                   │ 2. EXPLICIT MEMORY (Conscious)│
├─────────────────────────────────────────────────────────────┼───────────────────────────────┤
│ • Continuous background associative resonance.              │ • Deliberate tool call:       │
│ • Populates Tier 3 (128 slots in Layers 16–47).             │   `<|tool_call|>recall(...)`. │
│ • Evaluates cosine salience against current hidden state.   │ • Direct-to-layer latent KV   │
│ • Injects dense state diffs directly into KV cache.         │   rehydration across 48 layers│
│ • Zero token generation cost, zero prompt clutter.          │ • Zero-FLOP memory blit.      │
│ • Acts like intuition / background contextual familiarity.  │ • Model consciously reasons   │
│                                                             │   over memories in `<|think|>`.│
└─────────────────────────────────────────────────────────────┴───────────────────────────────┘
```

#### 1. Implicit Memory (Subconscious Working Memory Priming)
* Runs continuously on every conversational turn.
* Computes cosine similarity between the current Layer 47 hidden state and all stored centroid diffs.
* Populates the Top-128 most resonant diffs into the dedicated Tier 3 slots of the upper reasoning layers (Layers 16–47).
* Requires zero prompt manipulation and zero token generation overhead.

#### 2. Explicit Memory (Direct-to-Layer Latent Rehydration)
When the model requires deliberate, deep recollection of past sessions, tool logs, or codebases, it issues an explicit tool call (`recall`).

* **Elimination of the Text Re-Encoding Penalty:** Standard RAG systems serialize text to disk and re-inject it into Layer 0 as prompt tokens, incurring massive matrix FLOP penalties ($\sim 500\text{--}1500\text{ms}$ of batched GEMM) and destroying all high-dimensional latent activations formed during the original experience.
* **Direct Latent Rehydration:** Our engine stores the pre-RoPE $(K_l, V_l)$ tensor slices across all 48 layers on disk. When `recall` matches an episode, the engine performs a direct memory-mapped copy (`memcpy` / GPU UMA blit) of the pre-computed $(K_l, V_l)$ slices straight into the active sliding ring slots.
* **Zero Matrix Prefill FLOPs:** Rehydrating a 256-token episode takes $\approx \mathbf{0.05\text{ms}}$ (limited only by NVMe bus bandwidth), restoring 100% of the original latent nuance instantaneously.
* **Lightweight Control Frame:** The tool returns a structured JSON receipt informing the model's high-level thought process of the memory match, temporal metadata, and continuation token:
  ```json
  <|tool_response>{
    "status": "rehydrated",
    "memory_id": 1042,
    "tokens_restored": 256,
    "elapsed_time": "3 hours ago",
    "original_timestamp": 1740441600000,
    "original_clock": 1420,
    "summary": "Postgres database schema migration",
    "continuation_token": "tok_1043"
  }<tool_response|>
  ```

#### 3. Dual-Track Temporal Perception (Resolving RoPE Phase-Decoherence)
To resolve the fundamental duality between *when a memory is recalled* and *when it originally occurred*:
* **Track 1 (Operational Attention / Local RoPE Clock):** Memories are rehydrated into the local working memory coordinate frame ($t_{\text{current}}$). This prevents high-frequency RoPE phase-decoherence over large time deltas ($\Delta t \gg 10{,}000$), ensuring self-attention dot products remain mathematically sharp and legible.
* **Track 2 (Episodic Grounding / Explicit Temporal Awareness):** The tool control receipt provides exact epoch timestamps (`original_timestamp`, `elapsed_time`, `original_clock`), allowing the model's `<|think|>` channel to perform explicit, accurate temporal reasoning.

### C. The Memory Diff & Telemetry Header
Every serialized state delta emitted by any layer carries a 32-byte telemetry header for cost, vitality, and temporal tracking:

```zig
pub const MemoryDiff = extern struct {
    timestamp: u64,          // Token clock creation tick (RoPE coordinate)
    last_accessed: u64,      // Token clock last recall tick
    access_count: u32,       // Number of times recalled into working memory
    salience_norm: f32,      // Initial directional magnitude ||Δ||
    prediction_error: f32,   // Surprise / entropy delta when generated
    layer_id: u8,            // Originating layer depth (0..47)
    reserved: [7]u8,         // Alignment padding / future reward telemetry
    vector: [3840]f16,       // The 3840-D state delta vector (Gemma 4 12B)
};
```

### D. Cost & Telemetry Tracker
```zig
pub const LayerTelemetry = struct {
    cycle_count: u64,
    flops_executed: u64,
    flops_saved_by_quiescence: u64,
    active_landmark_slots: u16,
    repetition_heat_counter: u32,
};
```

---

## 3. PHASE 1: Zero-Retraining Streaming Engine & Dynamic Ring Baseline

### Objective
Build a standalone, single-binary Zig inference engine capable of running pre-trained Gemma/LLaMA weights with fixed-size layer memory and zero-copy streaming, validating output fidelity against standard baseline runners (like `llama.cpp`).

```
┌────────────────────────────────────────────────────────────────────────┐
│ PHASE 1 ARCHITECTURE                                                   │
│                                                                        │
│ Ingress Stream ──► [ Sliding FIFO Ring ] ──► [ Layer Forward Pass ]    │
│                           ▲                           │                │
│                           │                           ▼                │
│                  [ Static Anchors ]           [ Autoregressive Token ] │
│                  [ + FIFO Diffs   ]                                    │
└────────────────────────────────────────────────────────────────────────┘
```

### Technical Scope & Components
1. **Model Loader & Weight Mapper (`src/loader.zig`):**
   * Memory-map GGUF / `.safetensors` files directly into unified memory using `std.posix.mmap`.
   * Bind static pointers to weight matrices ($W_E, W_Q, W_K, W_V, W_O, W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}, W_U$).
2. **Tokenizer Engine (`src/tokenizer.zig`):**
   * Fast Byte-Pair Encoding (BPE) parser and vocabulary lookup table.
3. **Phase 1 Forward Pass Engine (`src/engine.zig`):**
   * **RMSNorm:** Hardware SIMD vector normalization.
   * **Linear Projections:** Optimized GEMV / GEMM kernels (CPU SIMD and Vulkan Compute shader fallback).
   * **RoPE:** In-place 2D plane rotations using monotonic token clocks.
   * **Attention & SwiGLU:** Scaled dot-product attention + SwiGLU / GeGLU non-linear MLP.
   * **Dynamic Sliding Ring:** Circular write pointer (`head = (head + 1) % RING_SIZE`) with zero memory copies between input chunks.
4. **Phase 1 KV Cache Management:**
   * Pin a conservative landmark zone (e.g., 64 slots for system prompt and recent summary) while the remaining ~4,032 slots function as the sliding FIFO context.
5. **Validation Suite:**
   * Verify perplexity and token generation coherence against standard Gemma baseline outputs.

---

## 4. PHASE 2: Hierarchical Event-Driven Layer Quiescence & Dual-Score Memory

### Objective
Implement multi-rate layer execution, vector diffing, and the 3-tier memory injection engine with in-engine GEMV associative recall.

```
┌────────────────────────────────────────────────────────────────────────┐
│ PHASE 2 ARCHITECTURE                                                   │
│                                                                        │
│ Layer 47 (Slow / Abstract)  ◄── [ High-Entropy Delta ] ─── (Sparse)    │
│         ▲                                                    ▲         │
│         │ (Diffs)                                            │ (Diffs) │
│ Layer 0 (Fast / Sensory)    ◄── [ Continuous Ingress ] ────────────────┤
│         │                                                              │
│         ▼                                                              │
│ [ Diff Extraction ] ──► [ In-Engine GEMV Scan ] ──► [ 3-Tier Landmarks]│
│         │                     ▲                                        │
│         ▼                     │                                        │
│ [ NVMe Ring Buffer ] ─────────┘                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### Technical Scope & Components
1. **Vector Diffing & Gating Engine (`src/diff.zig`):**
   * Compute directional state velocity: $\Delta_{\text{state}} = x_{\text{retained}}^{(t)} - x_{\text{retained}}^{(t-1)}$.
   * Activity evaluation: If $\frac{1}{N}\sum \|\Delta_{\text{state}}\|^2 < \tau$, mark upper layer quiescent.
   * Skip upstream compute pipelines on quiescent cycles.
2. **Inter-Layer Translation Adapters (`src/adapter.zig`):**
   * Lightweight projection MLPs between asynchronous layers to translate sparse event deltas into expected continuous input manifolds.
3. **3-Tier Dual-Score Memory Injection (`src/memory.zig`):**
   * **Tier 1 (Dynamic Anchors):** Pinned system prompt, persona, and tool declarations dynamically sized to $N_{\text{sys}}$ ($\approx 384$ slots) locked permanently across all 48 layers.
   * **Tier 2 (Temporal FIFO Recency):** Continuous sliding ring context ($3{,}712$ slots on lower layers, $3{,}584$ slots on upper layers).
   * **Tier 3 (Associative Hippocampal Recall):** 128 dedicated slots on upper reasoning layers (16–47), populated with historical episodic diffs selected via fast in-engine GEMV cosine scan:
     $$\text{Salience}(M_i) = \alpha \cdot \cos(x_{\text{current}}, M_i) + \beta \cdot e^{-\lambda \Delta t} + \gamma \cdot \|\Delta_i\|$$
4. **Persistent NVMe Ring Buffer (`src/storage.zig`):**
   * Pre-allocated circular memory-mapped file for logging `MemoryDiff` structs asynchronously via Linux `io_uring`.
5. **Benchmarks:**
   * Measure FLOP savings and throughput multipliers on long continuous streams.

---

## 5. PHASE 3: Cost-Governed Synaptic Plasticity & Thermodynamic Consolidation

### Objective
Enable true on-the-fly learning and long-term memory maintenance by coupling thermodynamic forgetting with fast-weight synaptic baking.

```
┌────────────────────────────────────────────────────────────────────────┐
│ PHASE 3 ARCHITECTURE                                                   │
│                                                                        │
│ [ Recurrent Working Memory Tax ]                                       │
│                │                                                       │
│                ▼  (If Tax > Plasticity Risk)                           │
│ [ Synaptic Weight Consolidation ] ──► Updates W_mlp in Unified RAM    │
│                                       (Zero Context Cost Thereafter)   │
│                                                                        │
│ [ NVMe Diff Archive ] ──► [ Thermodynamic Sieve ] ──► [ Prune / Forget]│
└────────────────────────────────────────────────────────────────────────┘
```

### Technical Scope & Components
1. **Thermodynamic Forgetting Sieve (`src/pruning.zig`):**
   * Background garbage collection loop evaluating diff vitality:
     $$\text{Vitality}(M_i) = \|\Delta_i\| \times \text{AccessCount}_i \times e^{-\lambda (t_{\text{now}} - t_{\text{last\_accessed}})}$$
   * Automatically prune low-vitality diffs from NVMe storage to maintain a clean, high-density associative search space.
2. **Synaptic Weight Baking Engine (`src/plasticity.zig`):**
   * Implement in-place low-rank weight modulation for MLP blocks:
     $$W_{\text{effective}} = W_{\text{base}} + \Delta W$$
   * When a pattern's cumulative working memory tax exceeds the plasticity risk threshold, compile the associative landmark directly into $\Delta W$.
   * Eliminates the need to repeatedly load familiar facts into working memory slots.
3. **Cost/Reward Optimizer (`src/governor.zig`):**
   * Runtime governor balancing task performance against compute, memory, and plasticity costs.

---

## 6. Phase Implementation Matrix

| Capability | Phase 1 (Baseline) | Phase 2 (Hierarchical) | Phase 3 (Plasticity) |
| :--- | :--- | :--- | :--- |
| **Model Retraining Required** | **None (Zero)** | Minimal (Adapters only) | None (Delta weights) |
| **Layer Context Layout** | Dynamic Ring + Static/FIFO | Dynamic Ring + 3-Tier Landmarks | Dynamic Ring + Dynamic Plasticity |
| **Layer Execution Timing** | Synchronous all layers | Event-driven multi-rate skipping | Cost-governed dynamic scheduling |
| **Long-Term Memory** | Ephemeral rolling buffer | Persistent NVMe + GEMV Recall | Persistent + Thermodynamic Sieve |
| **Model Weights** | Static Read-Only | Static Read-Only | Dynamic in-place $\Delta W$ baking |
| **Primary Metric** | Baseline inference parity | $3\times$–$5\times$ stream throughput | Zero-shot continuous retention |
