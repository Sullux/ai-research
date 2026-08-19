# Model Selection: Gemma 4 12B Unified

## Executive Summary

For the development and validation of our **Streaming Hierarchical Architecture**, we have selected **Gemma 4 12B Unified** as the primary target foundation model. 

This model represents the optimal balance of reasoning capability, clean dense architecture, native multimodal streaming, and hardware efficiency for our target Framework Desktop environment (AMD Unified Memory Architecture with 96 GB allocated compute RAM).

---

## 1. Gemma 4 Lineup Comparative Analysis

| Feature / Model | Gemma 4 E2B | Gemma 4 E4B | **Gemma 4 12B Unified** | Gemma 4 26B A4B (MoE) | Gemma 4 31B Dense |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Total Parameters** | 2.3B (5.1B w/ PLE) | 4.5B (8.0B w/ PLE) | **11.95B** | 25.2B | 30.7B |
| **Active Parameters** | 2.3B | 4.5B | **11.95B** | 3.8B (8 of 128 experts) | 30.7B |
| **Layer Count** | 35 | 42 | **48** | 30 | 60 |
| **Architecture Type** | Dense + PLE | Dense + PLE | **Dense Unified (Encoder-Free)**| Mixture of Experts | Standard Dense |
| **Modalities** | Text, Image, Audio | Text, Image, Audio | **Text, Image, Audio** | Text, Image | Text, Image (No Audio)|
| **Encoder Design** | External Encoders | External Encoders | **Direct Linear Projections** | External Encoders | External Encoders |
| **MMLU Pro** | 60.0% | 69.4% | **77.2%** | 82.6% | 85.2% |
| **LiveCodeBench v6** | 44.0% | 52.0% | **72.0%** | 77.1% | 80.0% |
| **Vocabulary Size** | 262,144 | 262,144 | **262,144** | 262,144 | 262,144 |

---

## 2. Key Architectural Decision Drivers

### A. The "Unified" Encoder-Free Breakthrough
In standard multimodal architectures (and smaller Gemma E-models), processing audio and vision requires running heavy external neural networks (e.g., Vision Transformers / ViTs or Audio Conformers) before feeding intermediate representations to the LLM.

**Gemma 4 12B Unified eliminates external encoders entirely:**
* Raw audio waveforms and visual patches are projected **directly into the LLM's embedding space via lightweight linear projection layers**.
* All modalities are processed natively by the same 48 decoder transformer layers.
* **Impact on Custom Zig Engine:** Drastically simplifies our inference runtime. We only need to implement standard matrix-vector operations and a simple linear ingress projection, rather than porting complex external vision/audio neural pipelines.

### B. Standard Clean Dense Weights (No PLE or MoE Quirks)
* **Avoids Per-Layer Embeddings (PLE):** The E2B and E4B models utilize PLE tables to inflate parameter capacity for mobile devices. This non-standard memory layout complicates tensor indexing and memory mapping.
* **Avoids MoE Routing Overhead:** While the 26B A4B MoE is efficient, implementing dynamic gating, top-k expert dispatching, and scattered memory lookups across 128 expert matrices in Phase 1 adds unnecessary surface area.
* **Uniform 48-Layer Structure:** Gemma 4 12B provides a standard, predictable 48-layer stack ideal for our hierarchical event-driven layer experiments.

### C. Native Audio Streaming Capability
* Unlike the 31B Dense model—which drops audio support—the 12B Unified model natively processes audio streams.
* This allows us to test **true full-duplex voice streaming** directly inside our Phase 1 / Phase 2 streaming pipeline without bolting on an external ASR (Whisper) stack.

### D. Strong Reasoning Baseline
With a **77.2% MMLU Pro** and **72.0% LiveCodeBench** score, the 12B model possesses high-fidelity semantic and code understanding. It is capable of sophisticated in-context reasoning, tool use, and `<|think|>` reasoning trace generation.

---

## 3. Hardware & Memory Fit (Framework Desktop)

### Target Host Specs:
* **Memory Architecture:** AMD Unified Memory Architecture (UMA)
* **Allocated Compute Memory:** 96 GB DDR5 RAM
* **Storage:** PCIe 4.0 NVMe SSD (~7,000 MB/s sequential read)

### Quantization & Memory Footprint:

| Precision / Quantization | Memory Required | Remaining RAM for Ring Buffers & Diffs | Expected Token Speed |
| :--- | :--- | :--- | :--- |
| **`bfloat16` (Unquantized)** | ~24.0 GB | **~72.0 GB Headroom** | 15–25 tok/s |
| **`Q8_0` / `FP8`** | ~12.5 GB | **~83.5 GB Headroom** | 30–40 tok/s |
| **`Q4_K_M` (4-bit)** | ~7.5 GB | **~88.5 GB Headroom** | 45–60 tok/s |

### Architectural Implication:
Even running unquantized in full `bfloat16`, the model consumes only 25% of our allocated compute memory. Over **70 GB of unified DDR5 RAM remains available** to hold:
* 48-layer unified dynamic ring buffers (`[4096, 4096]` per layer $\approx 1.6\text{ GB}$).
* In-memory historical diff search archives (100,000+ diff vectors $\approx 800\text{ MB}$).
* OS disk cache and asynchronous `io_uring` direct I/O ring buffers.

---

## 4. Phase 1 Implementation Parameters

When configuring the Phase 1 Zig loader (`src/loader.zig`) and tensor allocations:

* **Model Family:** Gemma 4
* **Variant:** 12B Unified
* **Number of Layers ($N$):** 48
* **Hidden Dimension ($D$):** 4,096
* **Vocabulary Size ($V$):** 262,144
* **Attention Mechanism:** Multi-Head / Grouped-Query Attention with Rotary Position Embeddings (RoPE)
* **MLP Non-Linearity:** GeGLU / SwiGLU gated linear unit
* **Reasoning Mode:** Native `<|think|>` channel support
