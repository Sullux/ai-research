# Low-Level Binary Wire Protocol Specification (`API.md`)

## 1. Overview & Transport Model

The Streaming Hierarchical Inference Engine provides a native, low-overhead binary wire protocol designed for high-throughput, low-latency, full-duplex communication over **Standard Input / Standard Output (STDIN / STDOUT)** when the engine is launched in server mode:

```bash
./zig-out/bin/infer --serve [OPTIONS]
```

### Protocol Characteristics:
* **Transport:** Standard POSIX anonymous pipes (STDIN / STDOUT).
* **Byte Ordering:** Little-Endian for all multi-byte integers and IEEE 754 floating-point values.
* **Framing:** Fixed 16-byte message envelope followed by a variable-length payload.
* **Full-Duplex:** Host $\to$ Engine (Inbound) and Engine $\to$ Host (Outbound) frames operate concurrently and asynchronously without blocking or half-duplex turn-taking locks.
* **Zero Parsing Overhead:** Direct binary struct mapping in C / Zig / Node.js Buffers without JSON stringification overhead on hot paths.

---

## 2. Binary Message Frame Envelope

Every message transmitted in either direction begins with a fixed **16-byte Header**:

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                      Magic (0x53554C58)                       |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|          Version (1)          |           Msg ID              |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|            Opcode             |            Flags              |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                         Payload Length                        |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                        Payload Data...                        |
|                            (...)                              |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

### Field Definitions:

| Offset | Field | Type | Description |
| :--- | :--- | :--- | :--- |
| `0..3` | **`magic`** | `u32` | Protocol magic constant: `0x53554C58` (ASCII `'SULX'`). |
| `4..5` | **`version`** | `u16` | Protocol version (`0x0001`). |
| `6..7` | **`msg_id`** | `u16` | Host-assigned message / correlation ID for request-response pairing. |
| `8..9` | **`opcode`** | `u16` | Message operation code (see catalog below). |
| `10..11`| **`flags`** | `u16` | Bitfield flags (e.g. streaming continuation, turn boundaries). |
| `12..15`| **`payload_len`**| `u32` | Length in bytes of the payload immediately following the header ($0 \le N \le 67,108,864$). |

### Standard Header Flags (`flags` Bitfield):

| Bit | Name | Value | Description |
| :--- | :--- | :--- | :--- |
| 0 | `FLAG_END_OF_TURN` | `0x0001` | Signals the completion of a conversational or reasoning turn (`<turn|>`). |
| 1 | `FLAG_INTERRUPTED` | `0x0002` | Marks the message or episode as suspended/interrupted. |
| 2 | `FLAG_STREAM_CHUNK` | `0x0004` | Intermediate chunk in a multi-packet streaming transmission. |
| 3 | `FLAG_REPLAY` | `0x0008` | Identifies payload as explicitly recalled memory replay. |
| 4 | `FLAG_THINKING` | `0x0010` | Identifies emitted token as internal reasoning channel (`<channel>thought`). |

---

## 3. Protocol Opcode Catalog

### Inbound (Host $\to$ Engine) Opcodes: `0x0001 .. 0x00FF`

| Opcode | Name | Description |
| :--- | :--- | :--- |
| `0x0001` | **`OP_STREAM_INPUT`** | Stream raw text or pre-tokenized token blocks into the primary Layer 0 input pipeline. |
| `0x0002` | **`OP_ABORT`** | Administrative emergency brake: halts generation immediately; preserves staging state as `is_interrupted`. |
| `0x0003` | **`OP_MEM_QUERY`** | Explicit memory search (`keywords`, `fulltext`, temporal range, or pagination cursor). |
| `0x0004` | **`OP_SET_CONFIG`** | Configure per-turn parameters (thinking budget, temperature, top-k, quiescence threshold, stop tokens). |
| `0x0005` | **`OP_TOOL_RETURN`** | Return tool execution output back to the model following an `OP_TOOL_CALL`. |
| `0x0006` | **`OP_MEM_COMMIT`** | Force immediate consolidation of the active staging buffer into persistent long-term storage. |
| `0x000E` | **`OP_PING`** | Keepalive / round-trip health check. |
| `0x000F` | **`OP_SHUTDOWN`** | Gracefully flush persistent diff stores, release GPU resources, and terminate the process. |

---

### Outbound (Engine $\to$ Host) Opcodes: `0x0101 .. 0x01FF`

| Opcode | Name | Description |
| :--- | :--- | :--- |
| `0x0101` | **`OP_STREAM_TOKEN`** | Real-time generated token with token ID, chronological clock $t$, layer quiescence bitmask, and UTF-8 text. |
| `0x0102` | **`OP_MEM_RESPONSE`** | Results of an `OP_MEM_QUERY` returning matched episode metadata, timestamps, and scores. |
| `0x0103` | **`OP_TOOL_CALL`** | Model-generated tool call request (tool name + JSON arguments). |
| `0x0104` | **`OP_STATUS`** | Live engine telemetry (tok/s, active vs quiescent layer breakdown, ring buffer slot usage, VRAM/UMA). |
| `0x010E` | **`OP_PONG`** | Reply to `OP_PING`. |
| `0x01FF` | **`OP_ERROR`** | Structured error notification. |

---

## 4. Payload Specifications

### 4.1. `OP_STREAM_INPUT` (`0x0001`) — Inbound
Appends text or pre-tokenized tokens into the continuous Layer 0 ring buffer.

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|     Mode      |    Reserved   |         Token Count           |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                     Text / Token Stream Data                  |
|                              (...)                            |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```
* **`Mode` (`u8`):**
  * `0x00`: Raw UTF-8 string (engine runs tokenizer).
  * `0x01`: Array of 32-bit Token IDs (`[Token Count]u32`).
* **`Token Count` (`u16`):** Number of token IDs if `Mode == 0x01`, otherwise 0.
* **`Stream Data`:** UTF-8 bytes or binary `[Token Count]u32` array.

---

### 4.2. `OP_STREAM_TOKEN` (`0x0101`) — Outbound
Emitted synchronously for every autoregressively decoded token ($N=1$).

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                           Token ID                            |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                          Clock (t)                            |
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                   Active Layer Bitmask (0..31)                |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                   Active Layer Bitmask (32..63)               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|          Text Length          |      UTF-8 Token Slice...     |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+                               +
|                              (...)                            |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```
* **`Token ID` (`u32`):** Vocabulary token ID (0..262143).
* **`Clock (t)` (`u64`):** Monotonic token timestamp in the continuous stream.
* **`Active Layer Bitmask` (`u64`):** Bitmask indicating which layers executed densely vs which were skipped due to quiescence (Bit $i = 1$ indicates Layer $i$ executed; Bit $i = 0$ indicates Layer $i$ was bypassed).
* **`Text Length` (`u16`):** Length in bytes of the decoded UTF-8 string slice.
* **`UTF-8 Token Slice`:** Exact decoded text characters for this token.

---

### 4.3. `OP_SET_CONFIG` (`0x0004`) — Inbound
Sets granular per-turn decode parameters and thinking channel controls.

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                         Thinking Budget                       |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                         Temperature                           |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                            Top-P                              |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                     Quiescence Threshold                      |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|     Top-K     |   Stop Count  |      Reserved (16-bit)        |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                     Stop Token IDs ([Count]u32)...            |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```
* **`Thinking Budget` (`u32`):** Maximum tokens allowed in `<channel>thought` before forcing output response (0 = disabled thinking).
* **`Temperature` (`f32`):** Sampling temperature ($0.0 = \text{greedy argmax}$).
* **`Top-P` (`f32`):** Nucleus sampling threshold.
* **`Quiescence Threshold` (`f32`):** Velocity gating threshold ($0.0 = \text{100\% dense execution}$).
* **`Stop Token IDs`:** List of `u32` tokens triggering immediate end-of-turn.

---

### 4.4. `OP_MEM_QUERY` (`0x0003`) — Inbound
Initiates an explicit memory query across the long-term `DiffArchive`.

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|     Query Mode|     Top K     |          Cursor ID            |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                       Target Timestamp                        |
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                         Weight Vector                         |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                        Weight Temporal                        |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                        Weight Pinned                          |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|          Query Length         |       Query Text / Terms...   |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+                               +
|                              (...)                            |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```
* **`Query Mode` (`u8`):**
  * `0x00`: `keywords` (Fast embedding table lookup, $<0.1\text{ ms}$).
  * `0x01`: `fulltext` (1-pass contextual prefill, $1\text{ ms}$).
  * `0x02`: `temporal_walk` (Chronological walk from `Target Timestamp`).
  * `0x03`: `pinned` (Retrieve system task anchors).
* **`Top K` (`u8`):** Maximum memory episodes to return.
* **`Cursor ID` (`u16`):** Continuation token for paginating through deeper results.
* **`Target Timestamp` (`u64`):** Anchor timestamp for chronological walks.
* **`Weights` (`f32`):** Linear coefficients for composite salience scoring.

---

### 4.5. `OP_MEM_RESPONSE` (`0x0102`) — Outbound
Returns matched episodic memories and metadata.

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|  Result Count |    Reserved   |         Next Cursor ID        |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                 Array of [Result Count] Memory Records        |
|                              (...)                            |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

#### Memory Record Layout (24 bytes + variable summary string):
```
| Timestamp (u64) | Salience Score (f32) | Access Count (u32) | Flags (u16) | Text Len (u16) | Summary Text... |
```

---

### 4.6. `OP_TOOL_CALL` (`0x0103`) — Outbound
Emitted when the model generates a structured tool execution call.

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|         Call ID               |        Name Length            |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|       Tool Name (UTF-8)       |       Arguments JSON...       |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+                               +
|                              (...)                            |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

---

### 4.7. `OP_TOOL_RETURN` (`0x0005`) — Inbound
Host provides the result of a tool execution back to the model.

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|         Call ID               |         Status (0=OK)         |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                      Result Data (UTF-8 / JSON)               |
|                              (...)                            |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

---

### 4.8. `OP_STATUS` (`0x0104`) — Outbound Telemetry
Periodic telemetry frame reporting engine performance and resource states.

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                       Tokens Per Second                       |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|        Active Slots           |        Archived Diffs         |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                   Active Layer Bitmask (0..31)                |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                   Active Layer Bitmask (32..63)               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                        GPU VRAM Used (MB)                     |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

---

## 5. Interaction Patterns & State Machine

### Scenario A: Continuous Chat Turn with Thinking Channel
```
Host (Node.js TUI)                       Engine (Zig --serve)
       │                                         │
       ├─── OP_SET_CONFIG (ThinkingBudget=512) ──►│
       ├─── OP_STREAM_INPUT ("Write a parser") ──►│
       │                                         ├── (Prefill & Decode)
       │◄── OP_STREAM_TOKEN (FLAG_THINKING) ─────┤
       │◄── OP_STREAM_TOKEN (FLAG_THINKING) ─────┤
       │◄── OP_STREAM_TOKEN (Normal output) ─────┤
       │◄── OP_STREAM_TOKEN (FLAG_END_OF_TURN) ──┤
```

### Scenario B: Model-Directed Memory Query Tool Call
```
Host (Node.js TUI)                       Engine (Zig --serve)
       │                                         │
       │◄── OP_TOOL_CALL ("memory_query") ───────┤
       ├─── OP_MEM_QUERY (mode=keywords) ────────►│
       │◄── OP_MEM_RESPONSE (3 matches) ─────────┤
       ├─── OP_TOOL_RETURN (payload=recalled) ───►│ (Streamed into Layer 0)
       │◄── OP_STREAM_TOKEN ("Based on...") ─────┤
```

### Scenario C: Administrative Abort (`OP_ABORT`)
```
Host (Node.js TUI)                       Engine (Zig --serve)
       │                                         │
       │◄── OP_STREAM_TOKEN (Token 41) ──────────┤
       │◄── OP_STREAM_TOKEN (Token 42) ──────────┤
       ├─── OP_ABORT ───────────────────────────►│ (Stops decode instantly)
       │                                         ├── (Commits episode: is_interrupted=true)
       │◄── OP_STATUS (state=idle) ──────────────┤
```
