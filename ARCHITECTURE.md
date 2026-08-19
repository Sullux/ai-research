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
Each layer $l \in [0, N-1]$ manages a fixed-size contiguous buffer:

```
┌────────────────────────────────────────────────────────────────────────┐
│ Total Fixed Allocation: [4096 Tokens x 4096 Hidden Dimensions]         │
├───────────────────────────────┬────────────────────────────────────────┤
│ Sparse Memory Landmarks       │ Sliding FIFO Ring Buffer               │
│ (Injected Diffs & VQ Centroids│ (Continuous Streaming Tokens)          │
│ [Dynamic: 64 .. 512 slots]    │ [Dynamic: 3584 .. 4032 slots]          │
└───────────────────────────────┴────────────────────────────────────────┘
```

### B. The Memory Diff & Telemetry Header
Every serialized state delta emitted by any layer carries a 32-byte telemetry header for cost, vitality, and temporal tracking:

```zig
pub const MemoryDiff = extern struct {
    timestamp: u64,          // Token clock creation tick (RoPE coordinate)
    last_accessed: u64,      // Token clock last recall tick
    access_count: u32,       // Number of times recalled into working memory
    salience_norm: f32,      // Initial directional magnitude ||Δ||
    prediction_error: f32,   // Surprise / entropy delta when generated
    layer_id: u8,            // Originating layer depth (0..31)
    reserved: [7]u8,         // Alignment padding / future reward telemetry
    vector: [4096]f16,       // The 4096-D state delta vector
};
```

### C. Cost & Telemetry Tracker
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
│ Layer 31 (Slow / Abstract)  ◄── [ High-Entropy Delta ] ─── (Sparse)    │
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
   * **Tier 1 (Static Anchors):** Pinned system persona and core goals (32 slots).
   * **Tier 2 (Temporal FIFO Recency):** Immediate sequential diffs (128 slots).
   * **Tier 3 (Associative Resonance):** Historical diffs selected via fast in-engine GEMV cosine scan:
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
