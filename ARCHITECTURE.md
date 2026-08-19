# Technical Architecture & Implementation Blueprint

## 1. System & Hardware Target

* **Platform:** Linux (Ubuntu x86_64, Framework Desktop).
* **Architecture:** AMD Unified Memory Architecture (UMA) with 96 GB shared RAM allocated to integrated GPU/compute.
* **Storage:** PCIe 4.0 NVMe SSD (up to ~7,000 MB/s sequential read/write).
* **Primary Implementation Language:** **Zig** (leveraging zero-overhead memory control, explicit allocators, C interop, and first-class SIMD / `io_uring` support).
* **Target Model Family (Phase 1):** Gemma 2/4 & LLaMA 3 architectures (utilizing GGUF / `.safetensors` model weights).

---

## 2. Low-Level Memory & Tensor Layout

Each layer $l \in [0, N-1]$ manages a fixed-size contiguous buffer in fast memory:

```
┌────────────────────────────────────────────────────────────────────────┐
│ Total Fixed Allocation: [4096 Tokens x 4096 Hidden Dimensions]         │
├───────────────────────────────┬────────────────────────────────────────┤
│ Sparse Memory Landmarks       │ Sliding FIFO Ring Buffer               │
│ (Injected Diffs & VQ Centroids│ (Continuous Streaming Tokens)          │
│ [Dynamic: 64 .. 512 slots]    │ [Dynamic: 3584 .. 4032 slots]          │
└───────────────────────────────┴────────────────────────────────────────┘
```

### Buffer Roles & Mechanics
1. **Sliding FIFO Ring Buffer:**
   * Ingests incoming streaming vectors in-place using a circular pointer (`head = (head + batch_size) % RING_SIZE`).
   * **Zero `memcpy` overhead:** Continuous temporal context flows seamlessly without copying data between "previous" and "new" partitions.
2. **Sparse Memory Landmarks (3-Tier Dual-Score Layout):**
   * Holds high-salience vectors partitioned into three dynamic tiers:
     * **Static Anchors (e.g., 32 slots):** System prompt, active persona, and global directives (pinned).
     * **Temporal FIFO Recency (e.g., 128 slots):** Immediate sequence of recent layer diffs ensuring local narrative flow.
     * **Associative Resonance (e.g., 96 slots):** High-salience historical memories dynamically pulled via an in-engine GEMV cosine scan against the layer's historical diff buffer.
   * **Adaptive Sizing:** When no historical recall is triggered, the landmark zone shrinks (e.g., to 64 slots), granting ~4,032 slots to the sliding ring. When dense historical retrieval is active, the landmark zone expands (up to 512+ slots).

---

## 3. The Execution Cycle Pipeline

For each streaming tick:

```
                     ┌──────────────────────────────┐
                     │ External Ingress / Lower Out │
                     └──────────────┬───────────────┘
                                    │ (Streaming vectors)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 1. ZERO-COPY RING INGESTION & ROPE CLOCK ASSIGNMENT                   │
│    - Advance circular ring head pointer; write new vectors in-place    │
│    - Pin active/retrieved landmark centroids to landmark slots         │
│    - Assign true monotonic token timestamps to each active slot        │
├────────────────────────────────────────────────────────────────────────┤
│ 2. LAYER FORWARD PASS                                                  │
│    - RMSNorm(x)                                                        │
│    - Q, K, V Projections (W_Q, W_K, W_V)                              │
│    - RoPE 2D Plane Rotations using true timestamps                     │
│    - Attention Matrix & Causal Masking                                 │
│    - Value Aggregation & Output Projection (W_O)                       │
│    - First Residual Add (x = x + Δ_attn)                              │
│    - RMSNorm(x)                                                        │
│    - SwiGLU / GeGLU Expansion (W_gate, W_up) & SiLU/GELU Non-linearity │
│    - Down-Projection (W_down)                                          │
│    - Second Residual Add (x = x + Δ_mlp)                              │
├────────────────────────────────────────────────────────────────────────┤
│ 3. SPARSE LANDMARK SELECTION & DIFF EXTRACTION                         │
│    - Compute State Velocity across window: Δ_state = x_new - x_old     │
│    - Select sparse high-salience vectors (top-k entropy / ||Δ||)       │
│    - Output: 4 to 32 sparse landmark vectors (instead of dense chunks) │
│    - Noise Filter: If mean ||Δ_state|| < τ, mark layer quiescent       │
├────────────────────────────────────────────────────────────────────────┤
│ 4. MEMORY PERSISTENCE & UPSTREAM PROPAGATION                           │
│    - Asynchronously append sparse landmarks to NVMe ring buffer        │
│    - If not quiescent, pass sparse landmarks to Layer l+1              │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Key Architectural Mechanisms

### A. Vector Diffing & Noise Gating
The state delta is computed as a directional difference across feature dimensions:

$$\Delta_{\text{state}} = x_{\text{retained}}^{(t)} - x_{\text{retained}}^{(t-1)}$$

* **Threshold Metric:** Normalized Cosine Distance or Mean Squared Euclidean Norm:
  $$\text{Activity} = \frac{1}{1024} \sum_{i=0}^{1023} \| \Delta_{\text{state}}[i] \|_2^2$$
* If $\text{Activity} < \tau$ (e.g., $\tau = 0.05$), the layer suppresses upstream propagation. The higher layer's input for that cycle is skipped, saving compute.

### B. Vector Quantization (VQ) & In-Engine GEMV Associative Recall
To store and retrieve long histories without text-based RAG overhead:
* **Storage Compression:** Older state diffs are quantized using **Residual Vector Quantization (RVQ)** and logged to the NVMe ring buffer.
* **Dual-Score Salience Ranking:** Before each cycle, the engine scans the in-memory historical diff buffer ($M \in \mathbb{R}^{K \times 4096}$) using a single fast matrix-vector dot product (GEMV) against the current state vector $x_{\text{current}}$:
  $$\text{Salience}(M_i) = \alpha \cdot \cos(x_{\text{current}}, M_i) + \beta \cdot e^{-\lambda (t_{\text{now}} - t_i)} + \gamma \cdot \| \Delta_i \|$$
* **Near-Zero Latency:** Scanning 10,000 historical diffs (80 MB of floats) takes <2ms on DDR5 / integrated compute, running asynchronously in the background.
* **Centroid Consolidation:** When rehydrating large historical episodes, a fast k-means/centroid reduction algorithm collapses thousands of diff records into the top associative landmark slots.

### C. True Temporal RoPE Coordinates
Each vector carries a metadata struct in memory:
```zig
pub const TokenVector = struct {
    timestamp: u64, // Monotonic token clock / millisecond tick
    data: [4096]f16,
};
```
When rotating $Q$ and $K$:
$$\text{angle}(i) = \theta_i \cdot \text{vector.timestamp}$$
This eliminates positional index collision between newly arrived tokens and historical memories injected from past hours or days.

### D. NVMe Direct I/O Ring Buffer (`io_uring`)
* Memory diffs are written to a fixed-size pre-allocated circular memory-mapped log file on NVMe storage.
* Disk writes and reads execute asynchronously via Linux `io_uring` kernel submission queues, ensuring zero CPU blocking or GPU pipeline stalls.

---

## 5. Phased Development Roadmap

```
┌─────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: Zero-Retraining Baseline (Approach A)                          │
│ - Build custom single-binary Zig inference engine (Gemma / LLaMA).      │
│ - Implement fixed 4,096-token layer buffer (Dynamic Ring + Landmarks).  │
│ - Implement VQ Centroid Compression on KV cache for Injected Context.   │
│ - Validate output coherence against standard baseline runners.          │
├─────────────────────────────────────────────────────────────────────────┤
│ PHASE 2: Hierarchical Event-Driven Layer Quiescence                     │
│ - Implement Vector Diffing & threshold gating between layers.           │
│ - Add lightweight Inter-Layer Translation Adapters (MLP bridges).       │
│ - Measure compute savings from upper-layer skipping on long streams.    │
│ - Implement True Timestamp RoPE indexing across memory injections.      │
├─────────────────────────────────────────────────────────────────────────┤
│ PHASE 3: True In-Place Plasticity & Online Learning (Approach B)        │
│ - Implement persistent NVMe `io_uring` diff logging.                    │
│ - Implement Hebbian fast-weight modulation / micro-LoRA updates         │
│   directly into MLP weight delta matrices (W_effective = W_base + ΔW).  │
│ - Benchmark permanent factual retention without full fine-tuning.       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Phase 1 Architecture Decision Summary

| Component | Phase 1 Choice | Rationale |
| :--- | :--- | :--- |
| **Engine Architecture** | Pure Zig with Vulkan Compute / CPU SIMD fallback | Maximum control, zero dependencies, direct memory control. |
| **Model Format** | GGUF / `.safetensors` (Gemma-2B / LLaMA-3.2-1B/3B) | Readily available weights, quick iteration cycle. |
| **Context Strategy** | **Approach A (Hierarchical KV Compressor + VQ Centroids)** | Works out-of-the-box with pre-trained weights without retraining. |
| **Temporal Indexing** | True timestamp offsets passed to RoPE kernels | Preserves relative temporal distance for injected memory slots. |
| **Storage Backend** | Memory-mapped circular log file + Zig POSIX `mmap` | Simplest zero-copy baseline before full `io_uring` integration. |
