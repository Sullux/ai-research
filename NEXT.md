# Implementation Roadmap & Sequence of Milestones

This document records the strategic order of operations for completing the Streaming Hierarchical Inference Engine and its interactive ecosystem.

---

## 1. Performance Optimizations

### Recommended High-ROI Optimizations
- [x] **GPU-Side Top-64 Logit Softcapping & Reduction (Highest ROI ⭐⭐⭐):**
  - Apply $\tanh$ logit softcapping directly on GPU and implement a lightweight 2-pass GPU Top-64 reduction shader.
  - Transfer only 64 candidate token IDs and logits (512 bytes) to the host per step instead of copying the full 1.0 MB (262k floats) logits buffer.
  - Eliminates host CPU softcapping and 262k-entry heap insertion latency (~10–12 ms per token), boosting end-to-end decode throughput from $\approx 19.4\text{ tok/s}$ to $\approx 25.0\text{ tok/s}$.
- [x] **Fused QKV Projection Dispatch in Decode (High ROI ⭐⭐):**
  - Combine separate $Q$, $K$, and $V$ GEMV dispatches into a single fused QKV projection dispatch per layer (using `shaders/fused_qkv_q4.wgsl`).
  - Eliminates 96 Vulkan dispatches and 96 pipeline execution barriers per step, reducing raw GPU decode latency by $\approx 1.5\text{–}2.0\text{ ms}$ (reaching $\approx 26.5\text{–}27.0\text{ tok/s}$).
- [x] **2D Shared-Memory Block Tiling in Batch GEMM (High ROI for Prefill ⭐⭐):**
  - Introduce $16 \times 32$ / $32 \times 32$ 2D block tiling using workgroup shared memory (`var<workgroup>`) in `shaders/batch_gemm_q4.wgsl`.
  - Reuses loaded activation vectors across multiple weight columns, delivering a $2\times\text{–}3\times$ prefill speedup on multi-token prompts and multi-turn prefill without vendor-specific extensions.

### Approaches to Avoid (Low ROI / High Complexity)
- **Vendor-Specific Cooperative Matrix Extensions (`VK_KHR_cooperative_matrix`):** Avoid driver-specific SPIR-V cooperative matrix intrinsics. They introduce fragile driver dependencies, platform fragmentation across Vulkan implementations, and high maintenance overhead for negligible gain over clean workgroup shared-memory tiling.
- **Complex Multi-Queue Pipelining:** Avoid overlapping next-token command graph recording and execution across multiple concurrent Vulkan queues. It introduces subtle race condition risks, fence/semaphore synchronization hazards, and high cognitive load for a marginal theoretical latency reduction ($\le 0.5\text{ ms}$).

---

## 2. Memory File CLI & Storage Subsystem (`--memory` & Config)
- [x] **CLI & Config Integration:** Added `--memory <path>` CLI parameter (defaulting to `./.episodic.mem`) in `src/main.zig` and `"memoryPath": "./.episodic.mem"` in `tui/config.json`.
- [x] **Episodic Tensor Serialization:** Implemented in `src/storage.zig` (`PersistentDiffStore`) with 64-byte deterministic aligned `FileHeader` and `EpisodeHeader`, unit-normalized 3,840-D centroids, cognitive provenance DAG (`parent_episode_id`, `child_count`), and contiguous multi-layer FP16 $(K, V)$ tensor slabs.
- [x] **Eviction-Triggered & Turn Consolidation:** Implemented in `src/hippocampus.zig` (`Hippocampus` 64-token staging ring buffer with temporal velocity tracking, 6,000 ms debounce, and turn boundary commit).
- [x] **Storage Telemetry & Verification:** Validated binary layouts, aligned offsets, parent lineage DAG propagation, and persistence across engine restarts.

---

## 3. Implicit Memory Priming (Subconscious Cosine Resonance)
- [x] **Background Cosine Resonance Scan:** Implemented in `src/memory.zig` using the unified multi-factor salience equation: $\text{Salience}(M_i) = \alpha \cdot \cos(x, C_i) + \beta \cdot e^{-\lambda \Delta t} + \gamma \cdot \log(1 + \text{child\_count} + \text{access\_count}) + \delta \cdot \|\Delta x\|$.
- [x] **Tier 3 Asymmetric Injection:** Implemented `primeTier3` in `src/memory.zig` populating Top-128 resonant memory representations into upper reasoning layers ($L_{16}\dots L_{47}$).
- [x] **Zero-Cost Priming:** Zero-FLOP / $<0.15\text{ ms}$ resident RAM centroid scan with zero NVMe disk I/O.
- [x] **Zero-Copy GPU Sync:** Added `syncRecallToGpu` in `src/model/memory_inject.zig` for direct host-visible UMA buffer updates on AMD hardware.

---

## 4. Explicit Latent Recall & Dual-Track Temporal Tooling
- [x] **Canonical Tool Definition:** Verified `recall` tool definition and schema in `tui/lib/tools/index.js` and `tui/PROMPT_KERNEL.md`.
- [x] **Wire Protocol Handling:** Handled `OP_MEM_QUERY` in `src/server.zig` with keyword query embedding, resident centroid scanning, and access count reinforcement.
- [x] **Direct-to-Layer Latent Rehydration:** Implemented `rehydrateEpisode` in `src/storage.zig` transferring multi-layer FP16 $(K, V)$ slabs directly into `DynamicRingBuffer` slots.
- [x] **Multi-Turn Retrospective Stress Testing:** Validated in `src/test_memory_lifecycle.zig` across multi-turn conversation branching, interruption handling, cold restart, RAM archive loading, and latent KV rehydration.

---

## 5. Streaming, Quiescence & Memory Stress Testing
- [ ] **Large Codebase Ingestion Test:** Stream the entire repository/project codebase into the engine and request a structured ~1,000-word architectural summary.
- [ ] **Quiescence Magnitude Sweeps:** Benchmark throughput (tok/s) and evaluate generation quality across quiescence thresholds ($0.0001 \to 0.35$).
- [ ] **Multi-Turn Retrospective Recall:** Test long conversational streams with multi-hop explicit memory queries (`"What did we discuss earlier regarding X?"`, `"What happened right after that?"`).
- [ ] **Interruption & Resumption Validation:** Verify that interrupted half-formed thoughts remain accessible and seamlessly synthesizable in subsequent turns.

---

## 6. Real-Time Online Learning (Synaptic Weight Adaptation)
- [ ] Track recurrent memory activation diffs ($\sum \text{Working Memory Tax} > \text{Plasticity Risk Tax}$).
- [ ] Implement fast online micro-LoRA / Hebbian weight updates ($\Delta W$) into MLP projection layers.
- [ ] Verify permanent retention of consolidated facts with zero context slot consumption and zero retrieval latency.
