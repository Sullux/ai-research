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
- [ ] **CLI & Config Integration:** Add `--memory <path>` CLI parameter (defaulting to `./.episodic.mem`) in `src/main.zig` and add `"memoryPath": "./.episodic.mem"` in `tui/config.json`.
- [ ] **Episodic Tensor Serialization:** Update `src/storage.zig` (`PersistentDiffStore`) to serialize 48-layer pre-RoPE $(K_l, V_l)$ episodic blocks alongside the 32-byte `MemoryDiff` telemetry header.
- [ ] **Eviction-Triggered Archival:** Automatically stream compressed $(K_l, V_l)$ tensor state directly to the memory-mapped `.episodic.mem` file as context rolls off Tier 2.
- [ ] **Storage Telemetry & Verification:** Validate file header, binary diff alignments, and persistence across server restarts.

---

## 3. Implicit Memory Priming (Subconscious Cosine Resonance)
- [ ] **Background Cosine Resonance Scan:** On each conversational turn, compute fast GEMV cosine similarity between the current Layer 47 hidden state and stored episodic memory centroids.
- [ ] **Tier 3 Asymmetric Injection:** Populate the Top-128 most resonant latent diffs directly into Tier 3 slots of upper reasoning layers (16–47) prior to generation.
- [ ] **Zero-Cost Priming:** Verify that implicit priming occurs with zero token generation cost and zero prompt clutter.
- [ ] **Latency Benchmark:** Ensure background implicit resonance scan adds $< 0.1\text{ms}$ per conversational turn.

---

## 4. Explicit Latent Recall & Dual-Track Temporal Tooling
- [ ] **Canonical Tool Definition:** Update the `recall` tool definition in `tui/PROMPT_KERNEL.md` with full parameters (`query`, `mode`, `continuation_token`).
- [ ] **Wire Protocol Extensions:** Support explicit memory query and rehydration opcodes in `src/protocol.zig` and `src/server.zig`.
- [ ] **Direct-to-Layer Latent Rehydration:** Implement zero-FLOP memory-mapped copy (`memcpy` / GPU UMA blit) of pre-RoPE $(K_l, V_l)$ slices directly into active sliding ring slots.
- [ ] **RoPE Re-Anchoring:** Apply 2D rotary position rotations at current operational clock coordinates ($t_{\text{current}}$) to maintain mathematically sharp, decoherence-free self-attention.
- [ ] **Dual-Track Temporal Framing:** Return lightweight JSON control receipts containing exact epoch timestamps, original creation clocks, elapsed time, and continuation tokens to keep the `<|think|>` channel temporally grounded.
- [ ] **TUI Tool Integration:** Wire `recall` tool handler in `tui/lib/tools/index.js` and controller loop.

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
