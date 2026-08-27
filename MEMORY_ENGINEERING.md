# Memory Engineering & Subsystem Implementation Specification

## Overview & Technical Objectives

This document details the engineering specifications, low-level binary formats, memory-mapped I/O pipelines, and implementation milestones for **Phase 2: The Streaming Hierarchical Memory Subsystem**.

The system implements the dual-process cognitive architecture defined in [MEMORY.md](MEMORY.md):
1. **Implicit Subconscious Priming:** High-speed in-engine GEMV cosine resonance scan projecting current hidden states ($x_{\text{final}} \in \mathbb{R}^{3840}$) against historical memory centroids, populating Tier 3 peripheral slots in upper transformer layers ($L_{16} \dots L_{47}$) with zero token generation cost.
2. **Explicit Latent Rehydration:** Deliberate tool-triggered (`recall`) zero-FLOP memory-mapped copy of pre-RoPE multi-layer $(K_l, V_l)$ tensor slices straight into active GPU ring buffer slots, eliminating the quadratic matrix prefill FLOP penalty of standard text-based RAG.

---

## 1. Hardware Environment & Storage Geometry

* **Host Platform:** AMD Ryzen AI Max+ 395 with Radeon 8060S (`gfx1150` / RDNA 3.5 architecture).
* **Memory Model:** Unified Memory Architecture (UMA) with 96 GB shared physical RAM.
* **Storage Bus:** PCIe 4.0 NVMe SSD (sequential read throughput $\ge 7{,}000\text{ MB/s}$).
* **Target Model:** Gemma 4 12B Unified:
  * Hidden size: $H = 3840$
  * Layers: $N_{\text{layers}} = 48$
  * KV dimension per layer: $H_{\text{KV}} = 1024$ (8 heads $\times$ 128 head dim)
  * Attention configuration: 8 full-attention layers + 40 sliding-window layers ($W = 1024$)

### Tensor Memory Footprint Per Token:
* **Per-Token KV Activation:**
  $$48\text{ layers} \times 2 \,(K, V) \times 1024\text{ elements} \times 2\text{ bytes (FP16)} = 196.6\text{ KB per token}$$
* **Consolidated Episode Block (e.g., 64 tokens):**
  $$64\text{ tokens} \times 196.6\text{ KB} \approx 12.58\text{ MB per episode}$$
* **Storage Capacity:** A 1.0 GB `.episodic.mem` file holds $\approx 80$ full 64-token high-fidelity episodic blocks or $\approx 20{,}000$ sparse landmark centroids.

---

## 2. Binary File Format Specification (`.episodic.mem`)

The memory store uses a deterministic, 64-byte-aligned, zero-copy memory-mapped file structure.

```
┌────────────────────────────────────────────────────────────────────────┐
│ .episodic.mem BINARY FILE LAYOUT                                       │
├────────────────────────────────────────────────────────────────────────┤
│ 1. FileHeader (64 Bytes, Fixed Offset 0)                               │
├────────────────────────────────────────────────────────────────────────┤
│ 2. Centroid Vector Index Table [Capacity × (64B Meta + 7680B Vector)]  │
├────────────────────────────────────────────────────────────────────────┤
│ 3. Multi-Layer KV Tensor Slab Storage [Capacity × Variable Block Slab] │
└────────────────────────────────────────────────────────────────────────┘
```

### A. File Header (64 Bytes)
Located at byte offset `0x00000000`:

```zig
pub const FileHeader = extern struct {
    magic: [4]u8,            // 'E', 'M', 'E', 'M' (0x4D454D45)
    version: u32,          // 1
    file_flags: u32,       // Bit 0: FP16 tensors, Bit 1: Q8_0 tensors, Bit 2: Compressed
    num_layers: u32,       // 48
    hidden_size: u32,      // 3840
    kv_dim: u32,           // 1024
    max_episodes: u32,     // Maximum allocated episode count (e.g., 16384)
    total_episodes: u64,   // Total committed episodes written to date
    write_head: u64,       // Next circular append slot index
    model_id_hash: u64,    // 64-bit FNV-1a hash of model weight signature
    reserved: [16]u8,      // Future expansion / alignment padding
};
```

### B. Episode Metadata Header (64 Bytes)
Each stored episodic entry begins with a 64-byte fixed descriptor:

```zig
pub const EpisodeHeader = extern struct {
    episode_id: u64,         // Monotonic globally unique episode index
    parent_episode_id: u64,  // Lineage pointer to ancestor episode active during formation
    created_timestamp: u64,  // Real-world epoch timestamp (Date.now() ms)
    last_accessed: u64,      // Epoch timestamp of most recent recall/citation
    start_clock: u64,        // Monotonic token clock t at episode inception
    token_count: u32,        // Number of sequential token slots in KV slab (0 if slab pruned)
    access_count: u32,       // Total times recalled/cited into working memory
    child_count: u32,        // Number of subsequent episodes derived from this record
    salience_norm: f32,      // Directional activation magnitude ||Δx||
    continuation_token: u32, // Next predicted token ID after episode boundary
    flags: u16,              // Bit 0: is_interrupted, Bit 1: has_tool, Bit 2: pinned, Bit 3: slab_pruned
    summary_len: u16,        // Length of UTF-8 summary / keyword tag slice (≤ 256 B)
};
```

### C. Episodic Payload Alignment
Immediately following each `EpisodeHeader`:
1. **Centroid Semantic Vector:** `[3840]f16` ($7{,}680\text{ bytes}$) unit-normalized final hidden state vector for background GEMV cosine scanning.
2. **Text / Summary Tag:** `[summary_len]u8` UTF-8 slice (up to 256 bytes) describing the topic or tool invocation.
3. **Multi-Layer KV Tensor Slab:** Contiguous memory block holding the multi-layer activation matrices (if `flags & 0x08 == 0` and `token_count > 0`):
   $$\text{Shape: } [48\text{ layers}][2\text{ (K, V)}][\text{token\_count}][1024\text{ elements}] \times \text{sizeof(f16)}$$

---

## 3. In-Memory Staging, Centroid Caching & Storage Lifecycle

```
[ Active Autoregressive Decode Loop (GPU) ]
                      │
   (Per Token Forward Pass: x_final ∈ R^3840, KV_l ∈ GPU VRAM)
                      │
                      ▼
┌────────────────────────────────────────────────────────┐
│  Stage 1: Hippocampus Consolidation Staging Buffer     │
│  - In-memory circular staging ring (capacity: 64 toks) │
│  - Tracks state velocities: Δx = x_t - x_{t-1}         │
│  - Tracks active parent context (current_parent_id)    │
│  - Tracks elapsed activity time (debounce: 6,000 ms)   │
└─────────────────────┬──────────────────────────────────┘
                      │
    [ Trigger: <turn|> boundary OR ||Δx|| > threshold ]
                      │
                      ▼
┌────────────────────────────────────────────────────────┐
│  Stage 2: Background Non-Blocking Consolidation Pass   │
│  1. Compute unit-normalized centroid vector            │
│  2. Copy 48-layer (K, V) slices from GPU UMA buffer    │
│  3. Insert metadata & centroid into RAM DiffArchive    │
│  4. Increment parent_episode.child_count in RAM/Disk   │
│  5. Async-append binary record to memory-mapped NVMe   │
└────────────────────────────────────────────────────────┘
```

### 100% In-RAM Centroid Indexing (Zero NVMe Read Overhead for Subconscious Priming)
* **Resident Index:** All `EpisodeHeader` descriptors and $3{,}840$-D Centroid vectors are loaded into RAM upon initialization.
* **Footprint:** $10{,}000$ episodes require only **$\approx 78\text{ MB}$ of RAM** ($64\text{ B meta} + 7{,}680\text{ B vector} \approx 7.8\text{ KB/entry}$).
* **Zero Disk I/O:** Implicit subconscious priming scans operate 100% against in-memory RAM buffers ($< 0.15\text{ ms}$). The NVMe storage device is **never accessed** during normal conversational decoding, reserving drive bandwidth exclusively for explicit KV rehydration.

### Staging & Debounce Rules:
1. **Intra-Turn Transient Isolation:** Sub-tokens within an ongoing sentence are buffered in RAM without touching disk or stalling the GPU queue.
2. **Turn Boundary Commit:** Encountering `<turn|>` (token `106`) or reaching 64 buffered tokens triggers immediate background consolidation.
3. **Non-Destructive Abort (`OP_ABORT`):** If generation is halted mid-stream, the buffered tokens are committed with `flags |= 0x01` (`is_interrupted = true`), preserving partial cognitive reasoning trajectories.
4. **Reconsolidation & Slab Aging:** When an older memory is recalled and built upon, the newly created episode naturally encapsulates the synthesized context. Unreinforced older episodes (`child_count == 0`, `access_count == 0`) can have their heavy $12.5\text{ MB}$ KV slabs stripped (`flags |= 0x08`, `token_count = 0`), retaining only the $7.8\text{ KB}$ centroid index permanently in RAM.

---

## 4. Dual Retrieval Engine Implementation

### A. Implicit Memory Priming (Subconscious Cosine Resonance)
* **Trigger:** Invoked once per conversational turn immediately after prompt prefill.
* **Multi-Factor Salience Algorithm:**
  1. Compute holistic salience against all resident in-memory centroids in `DiffArchive`:
     $$\text{Salience}(M_i) = \alpha \cdot \cos(x_{\text{final}}, C_i) + \beta \cdot e^{-\lambda (t_{\text{now}} - t_{\text{last}})} + \gamma \cdot \log(1 + \text{child\_count}_i + \text{access\_count}_i) + \delta \cdot \|\Delta x_i\|$$
  2. Select Top-$K$ ($K \le 128$) highest-scoring memory entries.
  3. Blit the selected historical $(K_l, V_l)$ vectors into the dedicated Tier 3 slots of reasoning layers ($L_{16} \dots L_{47}$).
* **Performance Budget:** $< 0.15\text{ ms}$ on CPU/GPU SIMD across 10,000 in-memory centroids.

### B. Explicit Latent Recall (Conscious Rehydration)
* **Trigger:** Model emits tool call `<|channel>call\nrecall({"query": "...", "mode": "semantic"})\n<channel|>`.
* **Execution Flow:**
  1. Map query keywords or embedded query vector against `DiffArchive`.
  2. Locate target episode record in the memory-mapped `.episodic.mem` file.
  3. Perform a direct memory-mapped copy (`memcpy` / GPU UMA blit) of the pre-RoPE $(K_l, V_l)$ slices straight into the active sliding ring window $[N_{\text{sys}} \dots N_{\text{sys}} + N_{\text{tokens}}]$.
  4. Apply RoPE phase rotation at the current operational clock ($t_{\text{current}}$) to maintain mathematically sharp self-attention.
  5. Return lightweight JSON receipt into `<|channel>call`:
     ```json
     <|channel>call
     <|tool_response>{
       "status": "rehydrated",
       "episode_id": 1042,
       "tokens_restored": 64,
       "elapsed_time": "2 hours ago",
       "original_timestamp": 1740441600000,
       "summary": "Zig Vulkan compute shader optimization discussion"
     }<tool_response|>
     <channel|>
     ```

---

## 5. Protocol & Wire Schema Extensions

### A. Binary Frame Opcodes (`src/protocol.zig`)

| Opcode | Name | Direction | Description |
| :--- | :--- | :--- | :--- |
| `0x0003` | **`OP_MEM_QUERY`** | Inbound (Host $\to$ Engine) | Query memory store by keyword embedding or temporal range. |
| `0x0006` | **`OP_MEM_COMMIT`** | Inbound (Host $\to$ Engine) | Force immediate flush of staging buffer to NVMe storage. |
| `0x0105` | **`OP_MEM_RESPONSE`** | Outbound (Engine $\to$ Host) | Return matched episode IDs, timestamps, and confidence scores. |

### B. `OP_MEM_QUERY` Payload Structure (`0x0003`)
```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|  Query Mode   |    Top-K      |            Reserved           |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                     Start Timestamp (u64)                     |
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                      End Timestamp (u64)                      |
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                    Query Text Payload (UTF-8)                 |
|                              (...)                            |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```
* **`Query Mode` (`u8`):**
  * `0x00`: Keyword / token embedding similarity.
  * `0x01`: Latent vector cosine similarity.
  * `0x02`: Chronological temporal range scan.

---

## 6. Phase 2 Implementation Roadmap & Milestones

```
┌────────────────────────────────────────────────────────────────────────┐
│ PHASE 2 IMPLEMENTATION SEQUENCE                                        │
│                                                                        │
│ Milestone M1: Memory-Mapped Binary Storage Engine (`src/storage.zig`)  │
│               - Implement `PersistentDiffStore` for 48-layer KV slabs  │
│               - 64-byte aligned `FileHeader` and `EpisodeHeader`       │
│               - CLI flag: `--memory <path>` (defaults to .episodic.mem)│
│               - Config integration: `"memoryPath": "./.episodic.mem"`  │
│                                                                        │
│ Milestone M2: Hippocampus Staging & Debounce (`src/hippocampus.zig`)   │
│               - 64-token staging ring buffer with temporal velocity    │
│               - Non-blocking commit at turn boundaries (<turn|>)       │
│               - Non-destructive interruption tag (`is_interrupted`)    │
│                                                                        │
│ Milestone M3: Implicit Priming Engine (`src/memory.zig`)               │
│               - Multi-layer Tier 3 peripheral slot injection (L16..47) │
│               - GEMV cosine resonance scan (< 0.15 ms benchmark)       │
│               - Verification: zero token cost subconscious priming     │
│                                                                        │
│ Milestone M4: Explicit Recall & Latent Rehydration                     │
│               - Implement `recall` tool definition in prompt kernel    │
│               - Direct memory-mapped KV blit into GPU sliding window   │
│               - RoPE phase re-anchoring at local clock t_current       │
│               - Structured JSON receipt emission to <|think|> channel  │
│                                                                        │
│ Milestone M5: End-to-End Stress & Multi-Turn Retrospective Testing     │
│               - Long continuous streaming tests across restarts        │
│               - Multi-hop recall benchmarks ("What did we discuss...") │
│               - NVMe throughput & memory leak validation               │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Verification & Success Criteria

1. **Storage Subsystem Parity:**
   - Memory-mapped file creation, header validation, and clean reopen across server restarts.
   - Persistence test: Ensure committed episodes are 100% byte-identical after engine restart.
2. **Implicit Priming Latency:**
   - Background cosine scan across $\ge 1,000$ episodes must complete in $< 0.15\text{ ms}$ per turn.
3. **Explicit Rehydration Speed:**
   - 64-token multi-layer KV rehydration from NVMe to GPU must complete in $< 1.0\text{ ms}$ (a $> 500\times$ speedup over re-prefilling).
4. **Coherent Recall Generation:**
   - Model accurately cites historical facts and code snippets from prior turns using `recall` without hallucinating or corrupting active dialogue context.
