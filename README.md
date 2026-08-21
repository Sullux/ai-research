# Streaming Hierarchical Inference Engine (Gemma 4)

A high-performance, single-purpose continuous inference engine written in pure Zig for Google's Gemma 4 models (`gemma-4-E2B` and `gemma-4-12B-it`). 

The engine implements a **Streaming Hierarchical Architecture** designed for zero-allocation, continuous conversational streaming with an infinite-horizon dynamic ring buffer, associative long-term binary diff-memory recall, multi-scale temporal quiescence, and a hardware-tailored Vulkan 1.3 compute backend built specifically for the **AMD Ryzen AI Max+ 395 (Radeon 8060S / RDNA 3.5 / gfx1150)** integrated GPU on 256-bit LPDDR5X-8000 unified memory architecture (UMA).

---

## Architectural & Research Documentation

For detailed architectural principles, design specifications, and mathematical foundations, see:

- [PROBLEM.md](PROBLEM.md) – Problem space, continuous conversation requirements, memory limits, and cognitive constraints.
- [APPROACH.md](APPROACH.md) – Hierarchical temporal stream, delta tracking, associative memory recall, and quiescent layer skipping.
- [ARCHITECTURE.md](ARCHITECTURE.md) – Structural blueprints, 3-tier ring buffer layout, memory-mapped persistence, and full-duplex execution model.
- [MEMORY.md](MEMORY.md) – Dual-process cognitive memory architecture, 6-second debounce consolidation, non-destructive interruption, and foreground streaming recall.
- [MODEL_SELECTION.md](MODEL_SELECTION.md) – Evaluation and selection of Gemma 4 architectures (E2B and 12B-it), layer typologies, and tokenization formats.
- [ENGINEERING.md](ENGINEERING.md) – Implementation roadmap, GPU kernel execution guidelines, and project coding standards.
- [API.md](API.md) – Low-level binary wire protocol specification, STDIN/STDOUT framing, opcodes, and full-duplex interaction models.
- [NEXT.md](NEXT.md) – Strategic implementation roadmap and ordered sequence of milestones.

---

## CLI Reference

### Synopsis

```bash
./zig-out/bin/infer [OPTIONS] [PROMPT...]
```

If no prompt arguments are supplied on the command line, the engine enters an interactive, continuous prompt session where conversation context is maintained across multiple turns.

### Options & Switches

| Switch | Arguments | Default | Description |
| :--- | :--- | :--- | :--- |
| `-m`, `--model` | `<path>` | `../gemma-4-E2B` | Path to the model directory containing `config.json`, `tokenizer.json`, and `model.safetensors`. |
| `-n`, `--max-tokens` | `<N>` | `128` | Maximum number of new tokens to generate for the response. |
| `--gpu` | — | Disabled | Enables Vulkan 1.3 GPU compute dispatch in unquantized 16-bit bfloat16 (`BF16`). |
| `--q4` | — | Disabled | Enables GPU compute acceleration with on-the-fly **Q4_0** layer weight dequantization and **Q8_0** output vocabulary classification. |
| `--q8` | — | Disabled | Enables GPU compute acceleration with on-the-fly **Q8_0** (8-bit signed integer) weight dequantization across all layers. |
| `--quant` | `<q4\|q8\|none>` | `none` | Explicitly selects the GPU weight quantization mode. |
| `--bench` | — | Disabled | Executes single-batch GPU compute throughput and latency benchmarking. |
| `--anchors` | `<N>` | `32` | Number of immutable early context anchor slots reserved at the start of the ring buffer. |
| `--window` | `<N>` | `512` | Number of rolling active sliding-window slots in the dynamic ring buffer. |
| `--recall` | `<N>` | `96` | Number of dynamic recall slots injected by the associative long-term diff-memory subsystem (up to 128 max). |
| `--storage` | `<path>` | Disabled | Path to a persistent binary diff archive (`.bin`). Enables cross-session long-term memory persistence. |
| `--no-memory` | — | Enabled | Disables associative long-term diff-memory ingestion and recall injection. |
| `--quiescence` | — | Disabled | Enables hierarchical multi-scale temporal quiescence gating to skip upper transformer layers during low activation velocity. |

---

## Usage Examples

### 1. Interactive 12B GPU Chat (Q4_0 / Q8_0 Acceleration)
```bash
./zig-out/bin/infer --model ../gemma-4-12B-it --q4
```

### 2. Single-Prompt 12B Execution with Gemma 4 Chat Formatting
```bash
./zig-out/bin/infer --model ../gemma-4-12B-it --q4 --prompt "<|turn>user
What is the capital of France?<turn|>
<|turn>model
" --max-tokens 50
```

### 3. Persistent Long-Term Associative Memory Session
```bash
./zig-out/bin/infer --model ../gemma-4-12B-it --q4 --storage memory_vault.bin
```

### 4. Running CPU Inference on Gemma 4 E2B
```bash
./zig-out/bin/infer --model ../gemma-4-E2B --max-tokens 30 "The capital of France is"
```

---

## Development & Engineering Guide

### Project Directory Structure

```
ai-research/
├── build.zig                   # Zig 0.14+ build configuration
├── shaders/                    # High-performance WGSL compute shaders
│   ├── gemv_q4.wgsl            # 128-thread / 4-row unrolled Q4_0 matrix-vector multiplication
│   ├── gemv_q8.wgsl            # 128-thread / 4-row unrolled Q8_0 matrix-vector multiplication
│   ├── gemv_bf16.wgsl          # 512-byte burst BF16 matrix-vector multiplication
│   ├── fused_mlp_q4.wgsl       # Fused Gate + Up + GeGLU activation Q4_0 compute kernel
│   ├── decode_attn.wgsl        # Multi-head GQA decode attention with parallel tree reduction
│   ├── qkv_rope.wgsl           # Fused Q/K/V RMSNorm, partial RoPE, and V unit norm cache writer
│   ├── add_rmsnorm.wgsl        # 128-bit vectorized residual addition + RMSNorm kernel
│   └── rmsnorm.wgsl            # 128-bit vectorized RMSNorm kernel
├── src/
│   ├── main.zig                # CLI entry point, argument parsing, interactive loop, and timing
│   ├── quant.zig               # Multi-threaded Q8_0 and Q4_0 quantization routines
│   ├── ring_buffer.zig         # 3-tier [anchors | window | recall] dynamic ring buffer
│   ├── memory.zig              # Dual-score associative DiffArchive and KV cache store
│   ├── storage.zig             # Memory-mapped binary diff persistence (PersistentDiffStore)
│   ├── quiescence.zig          # Multi-scale temporal quiescence velocity tracking
│   ├── vq.zig                  # Bounded spherical Vector Quantization (8-codebook residual VQ)
│   ├── tokenizer.zig           # Fast SentencePiece / BPE vocabulary parser and decoder
│   ├── safetensors.zig         # Zero-copy memory-mapped SafeTensors parser
│   ├── kernels.zig             # Pure CPU SIMD mathematical kernels (RMSNorm, RoPE, Softmax)
│   ├── gpu.zig                 # GPU subsystem root module
│   ├── gpu/                    # Vulkan 1.3 compute implementation
│   │   ├── vk_api.zig          # Pure Zig Vulkan C ABI type definitions & function pointers
│   │   ├── context.zig         # Vulkan instance, physical device (RADV GFX1150), and queue initialization
│   │   ├── buffer.zig          # Zero-copy unified memory (UMA) host-visible buffer management
│   │   ├── pipeline.zig        # Compute pipeline layout & SPIR-V shader module wrapper
│   │   ├── descriptors.zig     # Static pre-allocated descriptor set manager
│   │   ├── kernels.zig         # High-level GPU compute engine & single-batch command recorder
│   │   ├── model_dispatch.zig  # Single-batch whole-token 48-layer dispatch scheduler
│   │   └── model_gpu.zig       # GPU weight allocation, quantization, and descriptor binding
│   └── model/                  # High-level transformer orchestration
│       ├── types.zig           # Configuration, layer weight, and scratch buffer definitions
│       ├── loader.zig          # SafeTensors weight loader and model instantiator
│       ├── forward.zig         # CPU reference forward pass and GPU token forward dispatcher
│       └── ple.zig             # Per-Layer Embedding (PLE) auxiliary residual signal logic
└── tools/
    └── compile_all_shaders.py  # Naga-based WGSL -> SPIR-V -> Zig bytecode compiler
```

### Compiling Compute Shaders

All compute shaders are written in WGSL (`shaders/*.wgsl`) for clear maintainability and compiled into spec-compliant SPIR-V bytecodes and Zig arrays using `naga-wasi-cli`:

```bash
python3 tools/compile_all_shaders.py
```

### Building the Project

```bash
# Debug build
zig build

# Optimized ReleaseFast build (recommended for maximum inference performance)
zig build -Doptimize=ReleaseFast
```

The resulting binary is produced at `./zig-out/bin/infer`.

### Running Tests

```bash
# Run unit test suite
zig test src/main.zig

# Run GPU test suites (validates Vulkan compute pipelines and dequantization)
zig run src/test_gpu_suite.zig -lc
zig run src/test_q4_exec.zig -lc
```

---

## Coding Standards & Architectural Constraints

1. **Pure Zig Implementation:** Zero external C wrapper dependencies. Vulkan 1.3 ABI definitions are declared natively in Zig.
2. **File Size Guideline:** Keep code modules short and focused. All source files strictly adhere to the project constraint of `<200` lines per file.
3. **No Opinionated Chat Wrappers:** The inference engine processes exact token sequences and prompt inputs without hardcoded assumptions, allowing external frontends and TUIs full conversational flexibility.
