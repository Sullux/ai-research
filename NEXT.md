# Implementation Roadmap & Sequence of Milestones

This document records the strategic order of operations for completing the Streaming Hierarchical Inference Engine and its interactive ecosystem.

---

## 1. 4,096-Slot Layer Buffer & Dynamic 3-Tier Geometry
- [ ] **Fixed Capacity & Dynamic Anchoring:** Update `src/ring_buffer.zig` to enforce strict 4,096-slot capacity per layer and dynamically lock initial system prompt ($N_{\text{sys}} \approx 384$) as immutable Tier 1 anchors.
- [ ] **Depth-Asymmetric Partitioning:**
  - Lower Layers (0–15): Tier 1 Dynamic Anchors ($N_{\text{sys}}$) $+$ Tier 2 Sliding Ring ($4096 - N_{\text{sys}}$) $+$ Tier 3 ($0$ slots).
  - Upper Layers (16–47): Tier 1 Dynamic Anchors ($N_{\text{sys}}$) $+$ Tier 2 Sliding Ring ($4096 - N_{\text{sys}} - 128$) $+$ Tier 3 ($128$ recall slots).
- [ ] **Confined Modulo Wrapping:** Ensure write head cycles strictly within Tier 2 $[N_{\text{sys}} \dots N_{\text{sys}} + W - 1]$ via $\text{slot} = N_{\text{sys}} + ((\text{clock} - N_{\text{sys}}) \pmod W)$, never corrupting anchors.
- [ ] **GPU Indirection & Kernel Alignment:** Align GPU KV cache descriptor allocations, push constants, and decode attention gather kernels (`decode_attn.wgsl`) to the 4,096-slot buffer bounds.
- [ ] **Multi-Turn Verification:** Verify multi-turn dialogue beyond token 544 without token repetition loops or prompt eviction.

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
