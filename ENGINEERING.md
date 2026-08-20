# Engineering Strategy & Implementation Plan

## Overview & Philosophy

To ensure rigorous engineering discipline, Phase 1 is divided into two distinct, sequential milestones:

1. **Step 1: Foundational Baseline Model Runner (Reference Engine)**
   * Build a clean, standalone, single-binary LLM runner in pure **Zig**.
   * Verify mathematical correctness, GPU compute dispatch, memory-mapping, and tokenizer parsing against standard baseline outputs (e.g., Hugging Face / `llama.cpp`).
   * Zero experimental logic in this step—strictly establish a rock-solid, high-performance execution baseline.
2. **Step 2: Streaming Architecture & Dynamic Ring Integration (Research Engine)**
   * Evolve the verified runner to incorporate our novel **Dynamic Unified Ring Buffer**, **Sparse Landmark Registers**, **VQ Centroid Memory**, and **Continuous Streaming I/O**.

---

## 1. Core Engineering Decisions

### A. Model Precision & Format: Unquantized 16-Bit (`bfloat16` / `float16`)
* **Decision:** We will use unquantized, vanilla 16-bit floating-point weights (`bfloat16` / `float16`) in standard `.safetensors` or unquantized GGUF format.
* **Rationale:** 
  * With **96 GB of shared compute memory**, we have ample headroom to load the full unquantized weights of Gemma 4 12B (~24 GB) while maintaining ~72 GB of free RAM.
  * Avoids quantization artifacts, dequantization overhead in shaders, and loss of numerical precision during initial development.
  * Quantization (`Q8_0`, `Q4_K_M`, `FP8`) will be treated as an orthogonal, trivial optimization to be added after research validation.

### B. Micro-Model Bootstrapping for Rapid Iteration
* **Decision:** We will bootstrap and debug the Zig engine using a lightweight sibling model (e.g., Gemma 4 E2B / small test checkpoints) before scaling to the primary target (**Gemma 4 12B Unified**).
* **Rationale:**
  * Sub-second compile-run-test iteration cycles during initial tokenizer and tensor kernel debugging.
  * Once the math, memory mapping, and Vulkan shaders pass parity tests on the small model, switching to 12B Unified requires only updating file paths and tensor configuration constants.

### C. Modular State Interface (`ContextBuffer`)
* **Decision:** Decouple the attention computation from the underlying memory backing via a modular state interface.
* **Rationale:**
  * In Step 1, `ContextBuffer` implements a standard linear KV cache.
  * In Step 2, `ContextBuffer` is replaced with the `DynamicRingBuffer` without modifying the core GEMM, Attention, or SwiGLU kernels.

```zig
pub const ContextBuffer = struct {
    // Abstract interface implemented by StandardKVCache (Step 1)
    // and DynamicRingBuffer (Step 2)
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        appendTokens: *const fn (ctx: *anyopaque, tokens: []const TokenVector) void,
        getKeysView: *const fn (ctx: *anyopaque, layer: usize) []const f16,
        getValuesView: *const fn (ctx: *anyopaque, layer: usize) []const f16,
        getRoPEClocks: *const fn (ctx: *anyopaque) []const u64,
    };
};
```

---

### D. Associative Memory Representation: Normalized Hidden States vs. Delta Vectors
* **Decision:** Associative memory vectors ($M_i$) in the in-memory archive are stored as unit-normalized final hidden states ($\text{RMSNorm}(x_{\text{final}})$), while state velocities ($\Delta = x_t - x_{t-1}$) are retained for salience norms ($\|\Delta\|$) and disk serialization.
* **Rationale:**
  * **Semantic Resonance:** Cosine similarity $\cos(x_{\text{current}}, M_i)$ computes directional alignment in activation/concept space. Projecting current hidden state against a historical state vector measures whether the network is thinking about a similar concept. Resonating against a differential velocity vector ($\Delta$) measures rate-of-change alignment, which does not capture static semantic associations.
  * **Layer KV Rehydration:** When recalling memory $M_i$ into layer KV slots via $W_K, W_V$, projecting a normalized hidden state vector mimics a standard token embedding passing through the network. Projecting a raw delta produces derivative KV features incompatible with standard attention dot products.
  * **Role of the Delta:** The delta vector $\Delta$ remains critical for *gating* (triggering memory commits on novel events), *salience weighting* (the $\gamma \|\Delta_i\|$ score bonus), and *sparse disk compression* (storing compressed diffs on NVMe in Phase 2.2).

### E. Scale-Invariant Landmark Gating & True Chronological RoPE
* **Decision:** Commit landmark states when consecutive normalized states diverge below a cosine similarity threshold ($\cos(x_t, x_{t-1}) < 0.98$), and RoPE-rotate recalled KV slots at their original token clock coordinates.
* **Rationale:**
  * Cosine similarity is scale- and dimension-invariant, operating identically across varying hidden dimensions ($H = 1536$ for Gemma 4 E2B vs. $H = 3840$ for Gemma 4 12B).
  * RoPE rotation at the original creation timestamp allows the attention mechanism to naturally compute relative temporal decay $\cos(\theta \cdot (t_{\text{now}} - t_{\text{event}}))$, preserving biological temporal perception.

---

## 2. Step 1: Foundational Baseline Runner Roadmap (Complete)

```
┌────────────────────────────────────────────────────────────────────────┐
│ STEP 1 EXECUTION STAGES                                                │
│                                                                        │
│ Stage 1.1: Project Scaffold & CLI Harness                              │
│            - Initialize Zig build system (`build.zig`)                 │
│            - Explicit memory allocators (`std.heap`)                   │
│                                                                        │
│ Stage 1.2: Binary Weight Loader (`src/loader.zig`)                     │
│            - Memory-map model file (`std.posix.mmap`)                  │
│            - Parse tensor metadata & bind static typed pointers        │
│                                                                        │
│ Stage 1.3: Fast BPE Tokenizer (`src/tokenizer.zig`)                    │
│            - Subword BPE parser & vocabulary hash map                  │
│            - Token-to-string & string-to-token encoding/decoding       │
│                                                                        │
│ Stage 1.4: CPU SIMD Mathematical Kernels (`src/kernels_cpu.zig`)       │
│            - RMSNorm (with Gemma gain offset formula)                  │
│            - RoPE 2D Plane Rotations                                   │
│            - GEMV / GEMM Matrix Multiplication (AVX2/AVX-512)          │
│            - GeGLU / SwiGLU Gated Multiplications                      │
│            - Softmax / Temperature Sampler                             │
│                                                                        │
│ Stage 1.5: End-to-End CPU Inference & Parity Check                     │
│            - Autoregressive generation loop                            │
│            - Validate output logits against reference runner           │
│                                                                        │
│ Stage 1.6: GPU Acceleration Pipeline (`src/vulkan.zig`)                │
│            - Vulkan compute pipeline / SPIR-V shader compilation       │
│            - Offload heavy GEMV & Attention reductions to GPU          │
│                                                                        │
│ Stage 1.7: Interactive Terminal Chat Loop                              │
│            - Full terminal chat interface with Gemma 4 12B Unified     │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Phase 2: Hierarchical Streaming & Associative Memory Roadmap

```
┌────────────────────────────────────────────────────────────────────────┐
│ PHASE 2 EXECUTION STAGES                                               │
│                                                                        │
│ Stage 2.1: 3-Tier Dual-Score Associative Memory Index [COMPLETE]       │
│            - In-memory bounded DiffArchive (`src/memory.zig`)          │
│            - 3-tier DynamicRingBuffer (`src/ring_buffer.zig`):         │
│                [Anchors | Sliding FIFO Window | Associative Recall]    │
│            - Injection engine (`src/model/memory_inject.zig`)          │
│            - Dual-score recall: α·cos + β·e^(-λΔt) + γ·‖Δ‖             │
│            - CLI controls: `--recall [slots]`, `--no-memory`           │
│                                                                        │
│ Stage 2.2: Persistent NVMe Ring Buffer & Binary Diff Bank [COMPLETE]  │
│            - Memory-mapped 32-byte header `DiffHeader` binary layout   │
│            - Fast mmap disk store & DiffArchive bi-directional sync    │
│            - CLI control: `--storage [file_path]`                      │
│            - Source: `src/storage.zig`                                 │
│                                                                        │
│ Stage 2.3: Hierarchical Multi-Scale Temporal Quiescence [COMPLETE]     │
│            - Lower layers (0..N/2) execute densely every step          │
│            - Upper layers evaluate state velocity (Δ/‖x‖); skip on Δ<τ │
│            - Forced cadence refresh interval to prevent drift          │
│            - CLI control: `--quiescence`                               │
│            - Source: `src/quiescence.zig`                              │
│                                                                        │
│ Stage 2.4: VQ Centroid Memory Compression [COMPLETE]                   │
│            - Spherical K-Means & online centroid codebook (`src/vq.zig`)│
│            - Cluster deep historical archives into landmark centroids  │
│            - Nearest-centroid query and least-salient eviction         │
│                                                                        │
│ Stage 2.5: Continuous Full-Duplex Streaming Interface [COMPLETE]       │
│            - Process-global monotonic token clocking across REPL turns │
│            - Non-resetting DynamicRingBuffer & persistent anchor flow  │
│            - Monotonic RoPE temporal distance alignment                │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Hardware-Tailored GPU Acceleration Plan (AMD RDNA 3.5 / Radeon 8060S)

### A. Target Hardware & Architecture
* **Target Device:** AMD Ryzen AI Max+ 395 with Radeon 8060S (`gfx1150` / RDNA 3.5 architecture).
* **Memory Architecture:** AMD Unified Memory Architecture (UMA) with 96 GB–128 GB shared physical RAM.
* **Driver Target:** Mesa RADV (`/lib/x86_64-linux-gnu/libvulkan_radeon.so`, `libvulkan.so.1`, Vulkan 1.3).
* **Zero-Copy UMA Strategy:** Allocate memory with `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT`. Model weights and dynamic ring buffers are directly accessible by both CPU and GPU without PCIe staging buffers or host-to-device copy overhead.

### B. Implementation Stages

```
┌────────────────────────────────────────────────────────────────────────┐
│ AMD GPU COMPUTE IMPLEMENTATION STAGES                                  │
│                                                                        │
│ Stage G1: Vulkan Compute Scaffold & Device Discovery [COMPLETE]        │
│           - Direct Vulkan 1.3 ABI bindings in pure Zig (`vk_api.zig`)  │
│           - Select AMD Radeon 8060S (`gfx1150`) compute queue          │
│           - Allocate UMA host-visible / device-local shared buffers    │
│           - Verification: Synthetic GPU vector-add & compute test      │
│           - Files: `src/gpu/vulkan.zig`, `src/gpu/context.zig`         │
│                                                                        │
│ Stage G2: RDNA 3.5 Optimized Compute Shaders (SPIR-V) [COMPLETE]       │
│           - Shader 1: `gemv_bf16_f32` (Wave32 parallel matrix-vector)  │
│           - Shader 2: `fused_swiglu` (SwiGLU activation GEMV)          │
│           - Shader 3: `vec_add` test kernel                            │
│           - Verification: Mathematical parity vs CPU kernels (< 1e-4)  │
│           - Files: `src/gpu/shaders.zig`, `src/gpu/kernels.zig`        │
│                                                                        │
│ Stage G3: Model Pipeline Binding & End-to-End Execution [COMPLETE]     │
│           - Direct UMA host-visible activation buffer allocation       │
│           - CLI control: `--gpu` (with automatic fallback to CPU)      │
│           - AVX-512 / SIMD vectorized CPU kernel acceleration          │
│           - Verification: Parity & tok/s on Gemma 4 E2B & 12B-it       │
│           - Files: `src/gpu/model_dispatch.zig`, `src/main.zig`        │
│                                                                        │
│ Stage G4: Vectorized Bursts & Load-Time Quantization [COMPLETE]        │
│           - 128-bit vectorized burst loads (`uvec4`) for BF16 GEMV     │
│           - Load-time block quantization: Q8_0 (36 B/blk) & Q4_0 (20 B)│
│           - Real-time GPU dequantizing GEMV SPIR-V compute kernels     │
│           - CLI controls: `--q8`, `--q4`, `--quant [q8|q4]`            │
│           - Files: `src/quant.zig`, `src/gpu/shaders_q8.zig`, `q4.zig` │
│                                                                        │
│ Stage G7: Dynamic GPU Layer Quiescence & Indirect Dispatch [COMPLETE] │
│           - On-device temporal velocity tracker (`quiescence_gate.wgsl`)│
│           - Autonomous hardware gating via `vkCmdDispatchIndirect`      │
│           - Zero CPU recording / 0 CPU sync during decode skips         │
│           - Configurable velocity threshold (`--quiescence-threshold`)  │
│           - Verified scaling up to 30.9 tok/s on skipped layer passes  │
│           - Files: `shaders/quiescence_gate.wgsl`, `src/gpu/*`         │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Architectural Memory Throughput & Quantization Analysis

### A. Sustained Bandwidth vs Physical Hardware Limit
* **Hardware Ceiling:** AMD Ryzen AI Max+ 395 (LPDDR5X-8000 256-bit bus) has a theoretical peak DRAM bandwidth of **256.0 GB/s**, with real-world achievable UMA sustained read bandwidth of **~180–200 GB/s**.
* **Memory Transferred Per Token ($N=1$ Autoregressive Decode):**
  * 48 Layers Q4_0 Weights: $10.66\text{ B parameters} \times 0.5625\text{ B/param} \approx 6.00\text{ GB}$
  * Output Vocabulary Classifier (Q8_0): $262,144 \times 3840 \times 1.125\text{ B/param} \approx 1.13\text{ GB}$
  * Norm weights, KV cache reads, scratch activations: $\approx 0.05\text{ GB}$
  * **Total DRAM Read Per Token:** $\approx 7.18\text{ GB}$
* **Measured Performance:** At **18.6 tok/s**, sustained DRAM throughput is $18.6 \times 7.18\text{ GB} = \mathbf{133.5\text{ GB/s}}$ (**~67–74% of usable physical bus saturation**).
* **Physical Hardware Ceiling:** At 100% bus utilization (180–200 GB/s), maximum unquiesced throughput is **~25–27 tok/s**.

---

### B. Critical Finding: Quantization Limits on Output Vocabulary Projection (`embed_tokens`)

> **EMPIRICAL EXPERIMENT (Reverted):**
> 
> * **Hypothesis:** Quantizing the $262,144 \times 3840$ output classifier (`embed_tokens` transposed) to **Q4_0** would reduce memory read from $1.13\text{ GB} \rightarrow 0.566\text{ GB}$ per token, potentially saving ~0.56 GB of memory bandwidth and adding ~1.5–2.0 tok/s.
> * **Experimental Result:** 
>   * While throughput marginally changed (from ~16.5 to ~17.1 tok/s), Q4_0 quantization caused severe precision loss across the massive 262,144-dimensional logit classification distribution.
>   * During long-context generation, subtle compression artifacts skewed the tail logit distribution, causing rogue multimodal and control tokens (e.g. `<audio|>`, stray hyphens) to cross the argmax threshold mid-sentence (e.g., *"Thanks to a<audio|> property called superposition..."*).
> * **Architectural Decision:** 
>   * Transformer layer weights $0..47$ are quantized to **Q4_0** ($6.00\text{ GB}$).
>   * The final classification projection layer is **strictly retained in high-precision Q8_0** ($1.13\text{ GB}$) when running with `--q4`.
>   * Input token embedding lookups remain unquantized in **BF16**.

---

### C. Critical Finding: Fused Multi-Matrix QKV Projection vs Separate GEMVs

> **EMPIRICAL EXPERIMENT (Reverted):**
> 
> * **Hypothesis:** Fusing $Q$, $K$, and $V$ weight matrix projections into a single combined kernel dispatch per layer would eliminate 96 command buffer dispatches and 96 barrier stalls per token, while reusing $X_{\text{norm}}$ in L1 cache.
> * **Experimental Result:** 
>   * Performance slightly regressed by **~0.2 tok/s** (from $18.6 \rightarrow 18.4\text{ tok/s}$).
>   * **Root Cause 1 (Zero DRAM Volume Change):** In $N=1$ decode, $X_{\text{norm}}$ is only $15.36\text{ KB}$, which was already 100% resident in L1/L2 cache during separate dispatches. Zero bytes of DRAM read traffic were saved.
>   * **Root Cause 2 (Register Pressure & Occupancy Drop):** Binding 3 separate buffers (`W_Q`, `W_K`, `W_V`) with dynamic row-to-matrix routing (`if (target_matrix == 0u) ...`) increased Vector General-Purpose Register (VGPR) pressure beyond 32 registers. On AMD RDNA 3.5 (`gfx1150`), this reduced SIMD wave occupancy from **8 waves down to 4 waves per SIMD**, reducing the GPU's ability to hide memory latency bubbles.
>   * **Root Cause 3 (Negligible Launch Overhead):** Pre-recorded static command buffers already reduced CPU submission overhead to $0\text{ µs}$, and GPU CP dispatch latency across 96 dispatches totaled $< 48\text{ µs}$ ($< 0.09\%$ of the $54\text{ ms}$ token time).
> * **Architectural Decision:** Retain separate single-descriptor Wave32 GEMV kernels for $Q$, $K$, and $V$ to preserve 8 waves/SIMD maximum occupancy and zero register spilling.

---

### D. Critical Finding: 4-Row Tiled Memory Transposition vs Sequential Q4_0

> **EMPIRICAL EXPERIMENT (Reverted):**
> 
> * **Hypothesis:** Grouping 4 rows into interleaved $(4\text{ rows} \times 32\text{ cols})$ tiles ($80\text{ B}$ per tile: 4 float scales + 16 packed weight words) would allow all 4 waves in a 128-thread workgroup to read from a single contiguous 80-byte burst window, eliminating inter-row stride divergence.
> * **Experimental Result:** 
>   * Performance regressed by **~0.9 tok/s** (from $18.4 \rightarrow 17.5\text{ tok/s}$).
>   * **Root Cause 1 (Asynchronous Wavefront Scheduling):** On RDNA 3.5, the 4 waves in a workgroup are scheduled independently across SIMD execution slots without lockstep execution. Wave 0 may be at tile $b+2$ while Wave 3 is at tile $b$, meaning they rarely fetch from the same cacheline simultaneously.
>   * **Root Cause 2 (Broken Hardware Stream Prefetching):** In sequential row-major layout, each wave streams linearly in tight **$20\text{ byte}$ steps**, which triggers the hardware L1 vector stream prefetcher. In the 4-row tiled layout, each individual wave jumped in **$80\text{ byte}$ strides** per iteration, breaking hardware prefetch stream detection and increasing DRAM stall latency.
> * **Architectural Decision:** Retain standard sequential row-major Q4_0 layout ($20\text{ B}$ linear blocks) to maximize hardware streaming prefetcher efficiency.

---

## 6. Optimization Opportunities Roadmap (Phase 3 Path to 25–35+ tok/s)

Having exhausted micro-architectural shader layouts and proven that dense 48-layer decoding is physically bandwidth-saturated (~145–186 GB/s), the remaining path to $\ge 25\text{–}35+\text{ tok/s}$ is reducing total DRAM transfer volume:

### 1. Dynamic GPU Layer Quiescence (Phase 3 Primary Engine)
* **Mechanism:** Using on-device activation velocity $\|\Delta x\|^2$ tracking (`shaders/quiescence_gate.wgsl`) combined with indirect GPU command execution (`vkCmdDispatchIndirect`), upper transformer layers dynamically evaluate entropy and skip execution during steady-state generation.
* **Impact:** Skipping 25–35% of upper layers on steady-state tokens drops memory read from **$7.86\text{ GB} \rightarrow \mathbf{4.8\text{–}5.2\text{ GB/token}}$**, driving throughput directly to **$\mathbf{28\text{–}35+\text{ tok/s}}$**.

### 2. Layer-to-Layer Fused Post-FFN Residual Accumulation
* **Mechanism:** Fuse the residual addition ($x = x + \text{mlp\_out}$) directly into the subsequent layer's input RMSNorm kernel, eliminating intermediate buffer writebacks to DRAM/UMA between layers.
* **Impact:** **+0.3 to +0.5 tok/s**.

---

## 7. Testing & Verification Criteria

| Stage | Verification Milestone | Success Criteria |
| :--- | :--- | :--- |
| **Loader** | Load unquantized weights into RAM | Tensor headers and byte alignments match specification 100%. |
| **Tokenizer** | Round-trip tokenization test | `decode(encode(text)) == text` across standard test strings. |
| **CPU Kernels** | Synthetic tensor validation | CPU matrix operations match mathematical ground truth to $1e^{-4}$. |
| **Baseline Parity** | Gemma 4 inference comparison | Logits on prompt *"Once upon a time"* match reference implementation. |
| **GPU Compute** | Vulkan shader integration | Token generation speed $\ge 25$ tok/s on AMD integrated UMA compute. |
| **Streaming Step 2**| 50,000+ continuous token stream | Sustained generation with zero memory growth and intact narrative anchor. |
