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

## 2. Low-Level Binary Wire Protocol Specification (`API.md`) [COMPLETED]
- [x] Define the binary wire format over **STDIN/STDOUT** (using `--serve` mode).
- [x] Adopt **Pure Opcode Protocol** with compact 16-byte fixed-header envelope:
  - `magic` (4 bytes: `0x53554C58` / `'SULX'`)
  - `version` (2 bytes: `1`)
  - `msg_id` (2 bytes)
  - `opcode` (2 bytes)
  - `reserved` (2 bytes: `0`)
  - `payload_len` (4 bytes)
- [x] Catalog protocol opcodes:
  - `OP_STREAM_INPUT` (`0x0001`): Multimodal streaming into primary Layer 0 pipeline (text, tokens, soft vectors, audio PCM, images, video).
  - `OP_ABORT` (`0x0002`): Administrative emergency brake; preserves uncommitted staging buffer and tags episode `is_interrupted`.
  - `OP_MEM_QUERY` (`0x0003`): Explicit memory search (`keywords`, `fulltext`, temporal anchor, continuation cursor).
  - `OP_SET_CONFIG` (`0x0004`): Granular runtime parameters (thinking budget, temperature, stop tokens, quiescence threshold).
  - `OP_TOOL_RETURN` (`0x0005`): Host response to model tool request.
  - `OP_MEM_COMMIT` (`0x0006`): Force immediate consolidation of staging buffer to NVMe storage.
  - `OP_PING` (`0x000E`) / `OP_SHUTDOWN` (`0x000F`).
  - `OP_STREAM_CONTENT` (`0x0101`): Outbound conversational response token (with token type, clock $t$, active layer mask, UTF-8 text).
  - `OP_STREAM_THOUGHT` (`0x0102`): Outbound reasoning / thought channel token (`<channel>thought`).
  - `OP_TURN_COMPLETE` (`0x0103`): Outbound turn conclusion (token count, duration ms, tok/s, stop reason).
  - `OP_TOOL_CALL` (`0x0104`): Model-initiated tool execution request.
  - `OP_MEM_RESPONSE` (`0x0105`): Stream-injection memory telemetry badge (injected count, tokens, timestamps, cursor).
  - `OP_STATUS` (`0x0106`): Engine telemetry (tok/s, active vs. quiescent layer breakdown, ring slots, VRAM).
  - `OP_PONG` (`0x010E`) / `OP_ERROR` (`0x01FF`).

---

## 3. Engine Binary Protocol & Real-Time Memory Implementation (Zig) [COMPLETED]
- [x] Implement binary frame serializer/deserializer and dispatch in `src/protocol.zig`.
- [x] Implement `--serve` binary transport loop in `src/server.zig` and `src/server_queue.zig` reading from STDIN and writing to STDOUT.
- [x] Implement Hippocampal debounce consolidation staging buffer in `src/hippocampus.zig`.
- [x] Connect `OP_ABORT` with atomic signaling to halt generation instantly and preserve uncommitted state as `is_interrupted`.
- [x] Wire `OP_MEM_QUERY` handling:
  - `keywords`: Direct unquantized embedding lookup from `embed_tokens`.
  - Associative multi-layer KV scanning across historical episodes.
- [x] Verify unit test suite (`zig build test`) and full binary protocol integration (`tools/test_server_protocol.js`, `tools/test_server_abort.js`) on both E2B and 12B-it models.

---

## 4. Interactive Node.js TUI Application (`@sullux/tui`) [COMPLETED]
- [x] Create base system prompt kernel in `PROMPT_KERNEL.md` defining model persona, dual-channel reasoning, explicit memory recall, and live terminal control plane.
- [x] Implement pure JavaScript TUI client harness in `tui/` with zero external dependencies (linking local `@sullux/tui`).
- [x] Built binary framing client (`tui/lib/client/index.js`, `tui/lib/protocol/framing.js`) communicating over STDIN/STDOUT binary pipe with `./zig-out/bin/infer --serve`.
- [x] Built Virtual Terminal Buffer (VTB) with in-memory 2D active viewport ($30 \times 100$) and 5,000-line scrollback history (`tui/lib/terminal/buffer.js`, `tui/lib/terminal/session.js`).
- [x] Implemented cognitive tool catalog & stream parser (`tui/lib/tools/index.js`, `tui/lib/tools/parser.js`):
  - `recall`: Explicit memory queries via `OP_MEM_QUERY`.
  - `terminal_write`, `terminal_read`, `terminal_search`, `terminal_key`, `terminal_reset`: Full virtual terminal control plane.
  - `set_timer`, `cancel_timer`: Asynchronous cognitive wake-up notifications (`<|notification|>TIMER: note<|end_notification|>`).
- [x] Built declarative double-buffered UI (`tui/view.yaml`, `tui/theme.yaml`) with split panels for Thinking Scratchpad, Dialogue Stream, Terminal Viewport, Input Bar, and Status Telemetry.
- [x] Verified unit test suite with 100% pass rate (`tui/test/*.test.js`) and confirmed all JS files are strictly under 100 lines.

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
