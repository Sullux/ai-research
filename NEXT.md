# Implementation Roadmap & Sequence of Milestones

This document records the strategic order of operations for completing the Streaming Hierarchical Inference Engine and its interactive ecosystem.

---

## 1. Document Architecture & Conceptual Foundations [COMPLETED]
- [x] **`MEMORY.md`**: Document the dual-process cognitive memory architecture:
  - Implicit peripheral resonance (Tier-3 ring buffer slots) vs. Explicit foreground mental replay (1,024-token primary stream).
  - Hippocampal 6-second debounce consolidation window and out-of-band persistent storage.
  - Non-destructive interruption preserving suspended cognitive threads (`is_interrupted`).
  - Dual query modalities (`keywords` direct embedding lookup vs. `fulltext` contextual prefill).
  - High-layer semantic targeting (Layer 47) and full-stack multi-layer KV rehydration.
  - Cognitive provenance tagging (`<|start_recalled_memory|> ... <|end_recalled_memory|>`).
  - Connection to Phase 4 synaptic weight plasticity ($\Delta W$).

---

## 2. Low-Level Binary Wire Protocol Specification (`API.md`)
- [ ] Define the binary wire format over **STDIN/STDOUT** (using `--serve` mode).
- [ ] Design compact, 16-byte fixed-header frame envelope:
  - `magic` (4 bytes: `0x53554C58` / `'SULX'`)
  - `version` (2 bytes)
  - `msg_id` (2 bytes)
  - `opcode` (2 bytes)
  - `flags` (2 bytes: e.g., stream continuation, interrupt tag, end-of-turn)
  - `payload_len` (4 bytes)
- [ ] Catalog protocol opcodes:
  - `OP_STREAM_INPUT`: Streaming input text / token blocks into primary Layer 0 pipeline.
  - `OP_STREAM_TOKEN`: Real-time output token emitted by the engine (with token ID, UTF-8 text, clock $t$).
  - `OP_ABORT`: Administrative emergency brake / halt of generation; preserves uncommitted staging buffer and tags episode `is_interrupted`.
  - `OP_MEM_QUERY`: Explicit memory query (`keywords`, `fulltext`, temporal anchor, continuation cursor).
  - `OP_MEM_RESPONSE`: Result packet returning matched memory metadata and token lengths.
  - `OP_CONFIG`: Granular per-turn parameters (thinking budget/temperature, stop tokens, quiescence threshold).
  - `OP_TOOL_CALL`: Model-initiated tool execution request.
  - `OP_TOOL_RETURN`: Host response to model tool request.
  - `OP_STATUS`: Engine telemetry (active vs. quiescent layer breakdown, tok/s, memory usage).
  - `OP_ERROR`: Structured error reporting.
- [ ] Specify asynchronous state machine transitions and full-duplex duplex semantics.

---

## 3. Engine Binary Protocol & Real-Time Memory Implementation (Zig)
- [ ] Implement `--serve` binary transport loop in Zig (`src/protocol.zig`, `src/main.zig`).
- [ ] Implement Hippocampal debounce consolidation staging buffer in `src/memory.zig`.
- [ ] Connect `OP_ABORT` to tag active episodes and preserve working context without discarding state.
- [ ] Wire `OP_MEM_QUERY` handling:
  - `keywords`: Direct unquantized embedding lookup from `embed_tokens`.
  - `fulltext`: 1-pass short forward prefill pass through model layers.
  - Ingestion of recalled episodes directly into the 1,024-token primary input stream with cognitive provenance tags.
- [ ] Verify unit test suite and binary loop throughput.

---

## 4. Interactive Node.js TUI Application (`@sullux/tui`)
- [ ] Create Node.js interactive client harness using the `@sullux/tui` library.
- [ ] Child process orchestration over STDIN/STDOUT binary pipe.
- [ ] Visual UI components:
  - Live streaming response window.
  - Real-time performance telemetry dashboard (tok/s, active vs. quiescent layer gauge, VRAM/UMA memory usage).
  - Memory inspector panel showing active anchors, sliding window, and recalled historical diffs.
  - Thinking channel / reasoning display toggle.
- [ ] Keyboard interactions:
  - Interactive full-duplex typing.
  - Instant interruption via hotkey (triggering `OP_INTERRUPT`).
  - Runtime adjustment of thinking depth and quiescence thresholds.

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
