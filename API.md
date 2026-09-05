# Low-Level Binary Wire Protocol Specification (`API.md`)

## 1. Overview & Transport Model

The Streaming Hierarchical Inference Engine provides a native, low-overhead binary wire protocol designed for high-throughput, low-latency, full-duplex communication over **Standard Input / Standard Output (STDIN / STDOUT)** when the engine is launched in server mode:

```bash
./zig-out/bin/infer --serve [OPTIONS]
```

### Core Design Principles:
* **Pure Opcode Protocol (Zero Header Flags):** Every event, channel stream, and boundary condition is represented by a discrete opcode. Host dispatch is a simple, direct `switch (frame.opcode)` with zero bitmask checks.
* **Native Multimodal Streaming:** Supports text, discrete tokens, continuous soft embeddings, raw/encoded images, audio waveforms, and video frames.
* **Stream-Injection Memory Model:** Memory queries act as in-engine stream injection side-effects; the host receives lightweight telemetry badges rather than raw vector records.
* **Transport:** Standard POSIX anonymous pipes (STDIN / STDOUT) with lifecycle bound directly to the parent process.
* **Byte Ordering:** Little-Endian for all multi-byte integers and IEEE 754 floating-point values.
* **Fixed 16-Byte Envelope:** Every frame begins with a uniform 16-byte header followed by an optional payload.
* **Full-Duplex:** Host $\to$ Engine (Inbound) and Engine $\to$ Host (Outbound) frames operate concurrently and asynchronously without turn-taking locks.

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
|            Opcode             |          Reserved (0)         |
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
| `6..7` | **`msg_id`** | `u16` | Correlation ID for matching requests and responses (or `0` for unsolicited events). |
| `8..9` | **`opcode`** | `u16` | Operation code specifying the exact message type and payload structure. |
| `10..11`| **`reserved`** | `u16` | Reserved for 32-bit alignment padding (set to `0x0000`). |
| `12..15`| **`payload_len`**| `u32` | Length in bytes of the payload immediately following the header ($0 \le N \le 67,108,864$). |

---

## 3. Protocol Opcode Catalog

### Inbound (Host $\to$ Engine) Opcodes: `0x0001 .. 0x00FF`

| Opcode | Name | Description |
| :--- | :--- | :--- |
| `0x0001` | **`OP_STREAM_INPUT`** | Stream text, tokens, continuous soft vectors, audio PCM, images, or video frames into Layer 0. |
| `0x0002` | **`OP_ABORT`** | Administrative emergency brake: halts decoding immediately; preserves active state as `is_interrupted`. |
| `0x0003` | **`OP_MEM_QUERY`** | Explicit memory search (`keywords`, `fulltext`, temporal range, or pagination cursor). |
| `0x0004` | **`OP_SET_CONFIG`** | Configure runtime parameters (thinking budget, temperature, quiescence threshold, stop tokens). |
| `0x0005` | **`OP_TOOL_RETURN`** | Return tool execution result back into the model stream. |
| `0x0006` | **`OP_MEM_COMMIT`** | Force immediate consolidation of staging buffer to NVMe storage. |
| `0x0007` | **`OP_SET_SYSTEM`** | Initialize and prefill session system prompt with instructions and abstract tool definitions (JSON). |
| `0x000E` | **`OP_PING`** | Keepalive / round-trip latency probe. |
| `0x000F` | **`OP_SHUTDOWN`** | Gracefully flush stores, release GPU memory, and exit. |

---

### Outbound (Engine $\to$ Host) Opcodes: `0x0101 .. 0x01FF`

| Opcode | Name | Description |
| :--- | :--- | :--- |
| `0x0101` | **`OP_STREAM_CONTENT`** | Generated conversational / assistant text token or discrete multimedia output token. |
| `0x0102` | **`OP_STREAM_THOUGHT`** | Internal reasoning / thought channel token (`<channel>thought`). |
| `0x0103` | **`OP_TURN_COMPLETE`** | Signals end of turn (`<turn|>`), returning token count, duration, and average tok/s. |
| `0x0104` | **`OP_TOOL_CALL`** | Model-generated tool call request (tool name + JSON arguments). |
| `0x0105` | **`OP_MEM_RESPONSE`** | Results of an `OP_MEM_QUERY` returning injected episode counts, timestamps, and cursor. |
| `0x0106` | **`OP_STATUS`** | Live engine telemetry (tok/s, active vs quiescent layer breakdown, ring slots, VRAM). |
| `0x010E` | **`OP_PONG`** | Reply to `OP_PING`. |
| `0x01FF` | **`OP_ERROR`** | Structured error notification. |

---

## 4. Payload Specifications

### 4.1. `OP_STREAM_INPUT` (`0x0001`) — Inbound Multimodal Stream
Ingests text, tokens, soft vectors, audio PCM, images, or video frames into the continuous Layer 0 ring buffer.

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|     Mode      |   Sub-Format  |            Param 1            |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|            Param 2            |            Param 3            |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                     Stream Payload Data...                    |
|                              (...)                            |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

#### Multimodal Mode Matrix:

| `Mode` | Modality | `Sub-Format` | `Param 1` | `Param 2` | `Param 3` | Payload Structure |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `0x00` | **Text** | `0` | `0` | `0` | `0` | Raw UTF-8 bytes (engine tokenizes). |
| `0x01` | **Tokens** | `0` | Token Count (`u16`) | `0` | `0` | `[Token Count]u32` array. |
| `0x02` | **Soft Vectors** | `0` | Vector Dim ($H$) | Vector Count | `0` | `[Count * Dim]f32` (SigLIP / Audio latents). |
| `0x03` | **Audio PCM** | Format (`0`=S16LE, `1`=F32) | Channels (`1`/`2`) | Sample Rate ($Hz$) | `0` | Raw PCM audio samples. |
| `0x04` | **Raw Image** | Format (`0`=RGB24, `1`=RGBA) | Width ($px$) | Height ($px$) | `0` | Raw pixel bitmap bytes. |
| `0x05` | **Encoded Image**| Format (`0`=JPEG, `1`=PNG) | `0` | `0` | `0` | Compressed image file bytes. |
| `0x06` | **Video Frame** | Format (`0`=RGB24, `1`=JPEG) | Width ($px$) | Height ($px$) | Frame Index | Raw or compressed frame payload. |

---

### 4.2. `OP_STREAM_CONTENT` (`0x0101`) & `OP_STREAM_THOUGHT` (`0x0102`) — Outbound
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
|   Token Type  |    Reserved   |          Text Length          |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                     UTF-8 Token Slice...                      |
|                              (...)                            |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```
* **`Token ID` (`u32`):** Vocabulary token ID (0..262143).
* **`Clock (t)` (`u64`):** Monotonic token timestamp in the continuous stream.
* **`Active Layer Bitmask` (`u64`):** Bitmask indicating dense execution vs. quiescent skip per layer (Bit $i = 1$ indicates Layer $i$ executed).
* **`Token Type` (`u8`):**
  * `0x00`: Text token (UTF-8 slice is populated).
  * `0x01`: Audio codec token.
  * `0x02`: Image generation token.
  * `0x03`: Structural / Control marker (`<turn|>`, `<channel>`).
* **`Text Length` (`u16`):** Length in bytes of the decoded UTF-8 string slice.
* **`UTF-8 Token Slice`:** Decoded characters for this token.

---

### 4.3. `OP_TURN_COMPLETE` (`0x0103`) — Outbound
Emitted when generation halts at a turn boundary (`<turn|>`), max tokens, or after `OP_ABORT`.

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                       Generated Tokens                        |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                        Elapsed Time (ms)                      |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                     Average Tokens / Second                   |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|   Stop Reason |                   Reserved                    |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```
* **`Stop Reason` (`u8`):**
  * `0x00`: End of Turn token (`<turn|>`).
  * `0x01`: Max tokens limit reached.
  * `0x02`: Administrative abort (`OP_ABORT`).
  * `0x03`: Tool call requested.
  * `0x04`: Elastic resting boundary yield (`STOP_ELASTIC_YIELD`).

---

### 4.4. `OP_SET_CONFIG` (`0x0004`) — Inbound
Sets per-turn decode parameters and thinking channel controls.

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

---

### 4.5. `OP_MEM_QUERY` (`0x0003`) — Inbound
Initiates an explicit memory search across the long-term `DiffArchive`. Found memories are injected directly into the primary stream as foreground mental replay.

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

---

### 4.6. `OP_MEM_RESPONSE` (`0x0105`) — Outbound Telemetry Acknowledgment
Returns lightweight memory injection telemetry to the host application (zero raw vectors sent over the wire).

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
| Injected Count|     Status    |         Next Cursor ID        |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                     Total Injected Tokens                     |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|           Array of [Injected Count] Timestamps (u64)...       |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```
* **`Injected Count` (`u8`):** Number of historical episodes injected into the context.
* **`Status` (`u8`):** `0x00 = OK`, `0x01 = NoMatchesFound`, `0x02 = ArchiveEmpty`.
* **`Next Cursor ID` (`u16`):** Continuation token for deeper paginated search.
* **`Total Injected Tokens` (`u32`):** Total token volume injected into Layer 0.
* **`Timestamps` (`[Injected Count]u64`):** Monotonic clocks of injected episodes (for UI badges).

---

### 4.7. `OP_TOOL_CALL` (`0x0104`) — Outbound
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

### 4.8. `OP_TOOL_RETURN` (`0x0005`) — Inbound
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

### 4.9. `OP_SET_SYSTEM` (`0x0007`) — Inbound
Initializes and prefills session system instructions and abstract tool definitions into the KV cache ring buffer at startup. Enables zero Time-to-First-Token latency on subsequent user interactions.

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                     System Configuration JSON...              |
|                              (...)                            |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```
* **Payload:** UTF-8 encoded JSON object specifying abstract system instructions and available tools:
  ```json
  {
    "instructions": "Operational Directives:\n...",
    "tools": [
      {
        "name": "read",
        "description": "Read bounded chunk from a file",
        "parameters": {
          "type": "object",
          "properties": {
            "path": { "type": "string", "description": "File path" }
          },
          "required": ["path"]
        }
      }
    ]
  }
  ```
The engine formats the system prompt per the active model family's canonical chat template (e.g. Gemma 4 `<|turn>system\n<|think|>\n...<|tool>...<turn|>\n`), encodes tokens into Tier-1 KV cache anchors, and transitions to `STATUS_IDLE`.

---

### 4.10. `OP_STATUS` (`0x0106`) — Outbound Telemetry
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

## 5. Interaction Patterns & Node.js Dispatch Example

### Clean Host Dispatch Model (Node.js `@sullux/tui`)

```typescript
function onFrame(header: Header, payload: Buffer) {
  switch (header.opcode) {
    case OP_STREAM_THOUGHT:
      tui.appendThinking(payload.toString('utf-8', 28));
      break;

    case OP_STREAM_CONTENT:
      const tokenType = payload.readUInt8(24);
      if (tokenType === 0) { // Text
        tui.appendContent(payload.toString('utf-8', 28));
      } else if (tokenType === 1) { // Audio Codec
        audioVocoder.push(payload.readUInt32LE(0));
      }
      break;

    case OP_TURN_COMPLETE:
      const totalTokens = payload.readUInt32LE(0);
      const elapsedMs = payload.readUInt32LE(4);
      const tokSec = payload.readFloatLE(8);
      tui.setStatus(`${totalTokens} tokens in ${elapsedMs}ms (${tokSec.toFixed(1)} tok/s)`);
      break;

    case OP_MEM_RESPONSE:
      const count = payload.readUInt8(0);
      const totalTokens = payload.readUInt32LE(4);
      tui.showMemoryBadge(`Injected ${count} memories (${totalTokens} tokens)`);
      break;

    case OP_TOOL_CALL:
      const callId = payload.readUInt16LE(0);
      const nameLen = payload.readUInt16LE(2);
      const name = payload.subarray(4, 4 + nameLen).toString('utf-8');
      const args = JSON.parse(payload.subarray(4 + nameLen).toString('utf-8'));
      handleToolCall(callId, name, args);
      break;

    case OP_STATUS:
      updateTelemetry(payload);
      break;

    case OP_ERROR:
      tui.showError(payload.toString('utf-8'));
      break;
  }
}
```
