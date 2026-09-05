# Cognitive Working State Snapshot & Warm Boot Specification

## Overview

In traditional stateless LLM inference, conversational context is re-supplied and re-encoded on every turn or application launch. In our continuous streaming transduction architecture, context is maintained progressively inside the model's physical 4,096-slot dynamic ring buffer ($K, V$ activations, clock counter, and attention-mass salience).

To eliminate amnesia across sessions and bypass the multi-second GPU prefill penalty on startup, the engine incorporates **Cognitive Working State Snapshotting**.

---

## 1. Physical Snapshot Architecture

The working snapshot persists:
1. **Header Metadata**: Magic bytes (`SNAP`), version, global clock position, Tier 1 anchor count, ingested token count, and turn boundaries.
2. **Ring Buffer Geometry & Salience**: Per-slot clock counters, active boolean flags, and accumulated multi-layer attention mass.
3. **Physical KV Cache Slabs**:
   - For all active slots across all 48 layers, the key ($K$) and value ($V$) activations are extracted directly from Vulkan `HOST_VISIBLE | HOST_COHERENT` UMA memory.
   - Slices are packed contiguously. Inactive slots are skipped, shrinking the file size from ~6.1 GB down to **~30–75 MB** for early turns and **~1.2 GB** for large contexts.
4. **Stream Write-Ahead Anchor**: The ULID / stream message ID of the exact `.stream.jsonl` event corresponding to the snapshot point.

### Binary Header Structure

```zig
pub const SnapshotHeader = extern struct {
    magic: [4]u8 = SNAPSHOT_MAGIC, // 'SNAP'
    version: u32 = SNAPSHOT_VERSION, // 1
    clock: u64,
    num_anchors: u32,
    num_layers: u32,
    kv_dim: u32,
    max_slots: u32,
    total_ingested: u64,
    stream_id_len: u16,
    reserved_pad: u16 = 0,
    stream_id: [64]u8,
    turn_boundary_count: u32,
    turn_boundaries: [128]u32,
};
```

---

## 2. Wire Protocol Opcodes

The server communicates snapshot commands over the binary wire protocol:

| Opcode | Hex | Direction | Payload | Description |
| :--- | :--- | :--- | :--- | :--- |
| `OP_SNAPSHOT_SAVE` | `0x0008` | Client $\to$ Server | `path_len: u16, path: []u8, stream_id_len: u16, stream_id: []u8` | Trigger atomic snapshot save to path |
| `OP_SNAPSHOT_LOAD` | `0x0009` | Client $\to$ Server | `path: []u8` | Load snapshot and restore ring state |
| `OP_SNAPSHOT_STATUS` | `0x0107` | Server $\to$ Client | `status: u8, res: u8, slots: u16, clock: u64, id_len: u16, stream_id: []u8` | Confirm snapshot save (`status=0`) or load (`status=1`) |

---

## 3. Checkpointing Lifecycle & Write-Ahead Log (WAL) Replay

### A. Automated Checkpointing
- **Quiescence Trigger**: When a turn completes (`STOP_END_OF_TURN`), a 5-second debounce timer is armed. If no user input arrives within 5 seconds, the TUI dispatches `OP_SNAPSHOT_SAVE`.
- **Atomic File Swapping**: The server writes to `.snapshot.bin.tmp` and renames to `.snapshot.bin`, preventing corrupt partial writes on system crash.
- **Clean Shutdown**: Hitting `Ctrl+Q` or closing the application triggers an immediate synchronous snapshot flush before closing child processes.

### B. Instant Warm Boot & Catch-Up Replay
- On boot, the client checks for `.snapshot.bin`.
- If present, the client issues `OP_SNAPSHOT_LOAD`.
- The engine restores in **< 100 ms**, completely skipping the ~11-second system pre-cache.
- **Crash Recovery**: If the application crashed while un-checkpointed turns existed in `.stream.jsonl`, the client locates the stream item matching `streamId`, identifies any subsequent user turns, and automatically replays them through the cognitive pipeline to synchronize the KV cache with the UI state.
