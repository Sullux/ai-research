# Continuous Streaming Transduction Architecture

## 1. Problem Statement: The Monolithic Generation Bottleneck

Traditional Large Language Model architectures treat text generation as a **monolithic document synthesis task**. A user submits an entire turn of input, and the model enters an uninterrupted, open-ended decoding loop producing hundreds or thousands of tokens before returning control.

This paradigm creates four fundamental systemic failures:

1. **Unchecked Verbosity & Attention Dilution**: Models are rewarded during RLHF/SFT for generating exhaustive, multi-paragraph essays. This fills the finite context window with redundant syntactic filler, accelerating context exhaustion and increasing memory bandwidth pressure.
2. **Zero In-Flight Steerability (No Barge-In)**: Because generation is a closed-loop batch operation, external systems cannot provide real-time sensory feedback (such as user keystrokes, terminal status updates, or compiler errors) without destructively canceling the entire generation.
3. **Fragile Working Memory & Catastrophic Context Falloff**: In long-running agent tasks, as generation proceeds linearly through a standard FIFO ring buffer, foundational goals and high-level architectural constraints ("macro-intents") are evicted at the exact same rate as disposable local scratchpad reasoning ("micro-thoughts").
4. **Front-Loaded, Inflexible Deliberation**: Models are forced to plan an entire multi-step solution in a massive opening reasoning monologue. If an unforeseen constraint or branch point arises 500 tokens later, the model cannot gracefully pause, deliberate locally, and resume without restarting its global thought chain.

This architecture redesigns model interaction from batch document generation into **Continuous Streaming Transduction**: an interleaved, real-time sensory-motor loop where the model produces discrete, elastic semantic bursts while continuously monitoring uncertainty, environmental cost, and temporal urgency.

---

## 2. Theoretical Foundations: The Action Gate (Entropy, Urgency, and Cost)

In human cognition and optimal decision theory (such as the **Drift-Diffusion Model** of evidence accumulation), the decision to transition from internal deliberation (thinking) to external action (speech/content) is governed by an **Action Gate**.

```
                           THE DYNAMIC ACTION GATE
      Deliberation
      Threshold (Θ)
           ▲
           │          /▲\   [Low Urgency / Cheap Compute]
           │         / │ \  ───► High Threshold (Requires extreme certainty)
           │        /  │  \
           │       /   │   \ [High Urgency / Rising Delay Penalty]
           │      /    │    \───► Collapsing Threshold (Forces immediate action)
           │     /     │     \
           └────┴──────┼──────┴──────────► Time / Deliberation Steps (t)
                       │
             [ Gate Triggers: Confidence Gain Rate × Cost of Error < Cost of Delay ]
```

### The Governing Triad

1. **Entropy ($H$) — Epistemic Uncertainty**:
   - Represents the dispersion of the next-token probability distribution:
     $$H(X) = -\sum_{i} P(x_i) \log P(x_i)$$
   - High entropy indicates a semantic branch point or unresolved ambiguity.
   - Low entropy indicates that semantic and grammatical momentum has collapsed onto a clear, decisive trajectory.

2. **Cost ($C$) — The Penalty Surface**:
   - **Cost of Error ($C_{\text{err}}$)**: The downstream penalty of outputting a premature, incorrect, or syntactically invalid statement (e.g., a flawed algorithm signature or broken code fence).
   - **Cost of Compute/Delay ($C_{\text{delay}}$)**: The energy, memory bandwidth, and user latency consumed by continuing to deliberate on the internal scratchpad.

3. **Urgency ($U$) — The Time Derivative of Cost**:
   - Urgency is the rate of increase of delay cost over time ($U = \frac{dC_{\text{delay}}}{dt}$).
   - As silence persists in an interactive session, urgency mounts monotonically, compressing the deliberation threshold $\Theta(t)$ and forcing the model to emit output.

### The Action Gate Transition Rule
The model remains in the **thinking channel** (scratchpad) as long as the expected marginal value of further deliberation exceeds the cost of delay:

$$\text{Marginal Confidence Gain} \times C_{\text{err}} > U(t)$$

When entropy collapses below the dynamic threshold $\Theta(t)$, the gate opens, committing an elastic burst of tokens to the **response channel** until local entropy spikes again at the next structural branch point.

---

## 3. Internal Cognitive Signals Beyond Shannon Entropy

While token-level Shannon entropy provides an immediate baseline of output dispersion, modern transformer decoders expose deeper, zero-overhead internal scalar signals that reveal cognitive confidence in real time:

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│                             INTERNAL COGNITIVE SIGNALS                            │
├──────────────────────────┬───────────────────────┬────────────────────────────────┤
│ Metric                   │ Mathematical Source   │ Cognitive Meaning              │
├──────────────────────────┼───────────────────────┼────────────────────────────────┤
│ 1. Logit Margin (Δℓ)     │ ℓ_top1 - ℓ_top2       │ High margin = decisive intent;  │
│                          │                       │ Narrow margin = fork in path.  │
├──────────────────────────┼───────────────────────┼────────────────────────────────┤
│ 2. Attention Dispersion  │ Head-averaged Softmax │ Sharp = focused context lookup;│
│    (Head Entropy)        │ Weight Entropy        │ Diffuse = searching/ambiguity. │
├──────────────────────────┼───────────────────────┼────────────────────────────────┤
│ 3. Hidden State          │ ||Δh^(L)|| / ||h^(L)||│ Low delta = converged state;   │
│    Quiescence            │ (Layer-to-layer drift)│ High turbulence = deliberation.│
├──────────────────────────┼───────────────────────┼────────────────────────────────┤
│ 4. Perplexity Gradient   │ d(PPL)/dt over window │ Falling = crystallized train;  │
│                          │                       │ Rising = wandering/stuck loop. │
└──────────────────────────┴───────────────────────┴────────────────────────────────┘
```

These metrics allow the streaming orchestrator and inference engine to detect natural cognitive resting points without relying on crude fixed token counts.

---

## 4. Hierarchical Intent Pinning vs. Ephemeral Scaffolding

When generation is continuous and interleaved (alternating between short thinking bursts and outward output), working memory exhibits a strong structural asymmetry:

$$\text{Permanent Concrete (Output)} \gg \text{Disposable Scaffolding (Micro-Thoughts)}$$

```
[ Thought Phase (High Entropy / Global) ]
💭 "Need binary search over rotated array. First find pivot. Then adjust low/high."
                               │
               (Entropy drops, Intent crystallizes)
                               ▼
[ Action Phase (Low Entropy / Concrete Output) ]
🤖 `def search(nums: list[int], target: int) -> int:`
   `    low, high = 0, len(nums) - 1`
                               │
        (Entropy rises: reached the complex while loop condition)
                               ▼
[ Thought Phase (Mid-Flight Local Scratchpad) ]
💭 "Inside while low <= high: check if left half is sorted (nums[low] <= nums[mid])."
                               │
               (Entropy drops, Intent crystallizes)
                               ▼
[ Action Phase (Concrete Output) ]
🤖 `    while low <= high:`
   `        mid = (low + high) // 2`
```

### The Eviction Hierarchy

1. **Ephemeral Scaffolding (Micro-Thoughts)**:
   - Once the concrete output tokens (`mid = (low + high) // 2`) are generated, the preceding micro-thought (*"inside while loop..."*) has fulfilled its causal purpose.
   - The emitted code itself now carries the semantic state forward. The micro-thought can be dissolved/evicted from the active KV cache first.
2. **Sequential Concrete (Older Output Tokens)**:
   - Output tokens that fall outside the local sliding window can be gradually evicted or summarized, relying on global structural anchors.
3. **Macro-Intent & Invariants (The Blueprint)**:
   - Foundational constraints (system mission, active plan goals, class definitions) must remain permanently accessible in memory regardless of how many thousands of output tokens are generated.

---

## 5. Universal Eviction: Dynamic Attention-Mass Accumulation (Heavy Hitter)

Rather than hand-crafting brittle rules to classify thoughts into arbitrary categories, the engine relies on the transformer's intrinsic self-attention mechanisms to vote on token survival.

### The Mathematics of Attention Mass
On every decode step $t$, the GPU attention kernel (`shaders/decode_attn.wgsl`) computes softmax attention weights $\alpha_{t, i}$ for all active slots $i \in [0 \dots S-1]$. The engine accumulates the total incoming attention mass for each slot:

$$\text{Salience}(i) = \sum_{t = t_{\text{insert}}}^{T} \alpha_{t, i}$$

- **High-Value Anchors & Active Macro-Intents**: Accumulate consistent attention mass from almost every subsequent generation step across the 8 full-attention layers. Their salience score climbs monotonically.
- **Transient Scaffolding & Syntactic Connectors**: Experience a sharp, brief attention spike during immediate emission and then flatline. Their salience score decays relative to the active working set.

```
[ Active Dynamic Ring Buffer (4,096 Physical Slots) ]
┌────────────────────────────────────────────────────────────────────────┐
│ [Tier 1: Fixed Anchors] │            [Tier 2: Dynamic Working Context]         │
│ Slots 0..511            │ Slots 512..4095                              │
│ - System Prompt         │ - Active Interleaved Thoughts & Output       │
│ - High-Salience Memory  │ - Monitored by Cumulative Attention Mass     │
└─────────────────────────┴──────────────────────────────────────────────┘
                                     │
                 [ Ring Buffer Approaches Full Capacity ]
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   GPU UMA DEFRAGMENTATION & COMPACTION                 │
│ 1. Scan candidate eviction window (oldest Tier 2 slots).               │
│ 2. Retain top-K heavy hitters (highest cumulative attention mass).     │
│ 3. Evict cold slots (ephemeral micro-thoughts and stale syntax).       │
│ 4. Compact active KV slabs in-place in VRAM (~1.3 ms on Ryzen AI Max+).│
└────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Cognitive Saturation & The Dual-Engine Architecture

### The Problem: Working Memory Saturation (Cognitive Load Cliff)
During large, complex engineering tasks (e.g. implementing a complete multi-tier subsystem or building a full multi-file application), the model generates dozens of guiding macro-thoughts and invariant architectural rules. Over hundreds of decode steps, all of these tokens accumulate high cumulative attention mass.

Eventually, the 4,096-slot physical ring buffer fills. However, when the compaction algorithm scans Tier 2, **even the lowest-salience candidate tokens possess substantial attention mass**. The disposable tail of transient micro-thoughts has completely vanished.

```
 HEALTHY WORKING MEMORY (High Gini)           SATURATED COGNITIVE STATE (Low Gini)
 Attention                                   Attention
 Mass                                        Mass
  ▲   █                                       ▲   ████████████████████████████████
  │   █ █                                     │   ████████████████████████████████
  │   █ █ █                                   │   ████████████████████████████████
  │   █ █ █ █ ▄ ▂                             │   ████████████████████████████████
  └────────────────────────► Slots            └────────────────────────► Slots
   [Clear targets for eviction:                [NO disposable tail:
    80% disposable scaffolding]                 Eviction causes amnesia / amorphic drift]
```

### Mathematical & Empirical Indicators of Saturation

The engine detects this condition in real time without fine-tuning via three complementary signals:

1. **Salience Floor Elevation ($S_{\min}$ / $S_{\text{p10}}$)**:
   - When scanning the oldest candidate eviction window, if the 10th percentile salience score exceeds a critical threshold ($S_{\text{p10}} > \Theta_{\text{sat}}$), no safe eviction candidate exists.
2. **The Attention Gini Coefficient ($G$)**:
   - Evaluates the dispersion of attention across all active Tier 2 slots:
     $$G = \frac{\sum_{i=1}^{n} \sum_{j=1}^{n} |\alpha_i - \alpha_j|}{2n \sum_{i=1}^{n} \alpha_i}$$
   - **$G \ge 0.65$ (Focused)**: Attention is concentrated on a small set of dominant anchors and recent tokens.
   - **$G < 0.35$ (Attention Starvation)**: Attention is diffuse and stretched thin across the entire ring buffer.
3. **Upper-Layer Attention Entropy ($H_{\text{attn}}$ in $L_{16}\dots L_{47}$)**:
   - In the 8 full-attention reasoning layers, softmax entropy spikes toward maximum ($\log S$), signifying that attention heads are thrashing across competing requirements.

---

### The "Legs vs. Wheels" Dilemma: Articulation vs. Latent Memory

When cognitive saturation occurs, humans and machines face two fundamentally different solutions:

1. **"Writing It Down" (Symbolic Articulation — The "Legs")**:
   - **Why humans write things down**: Writing forces an **autoregressive compression & disentanglement pass**. Vague, conflicting sub-goals that coexist ambiguously in latent space must be resolved into clean, discrete symbolic tokens.
   - **Cognitive Benefit**: Creating a structured `plan()` or outline reinforces clarity, resolves latent contradictions, and creates a crystal-clear prompt anchor that steers subsequent attention heads with minimal entropy.
   - **The Flaw**: It is lossy. You discard 95% of the mechanical code syntax, variable registers, and fine details.

2. **Latent Archival (Subconscious Snapshotting — The "Wheels")**:
   - **The Superpower**: Instantaneous, bit-exact snapshotting of 48-layer FP16 $(K, V)$ slabs into `.episodic.mem` without prompt bloat or token generation latency.
   - **The Flaw**: If the latent space is messy or entangled, snapshotting it preserves the confusion alongside the context.

---

### The Synthesis: The Dual-Engine Architecture

Rather than treating symbolic tool calls and latent archival as mutually exclusive, the architecture assigns each mechanism to its optimal cognitive tier:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        THE DUAL-ENGINE ARCHITECTURE                    │
├───────────────────────────────────┬────────────────────────────────────┤
│ EXECUTIVE META-COGNITION (Legs)   │ LATENT WORKING SUBSTRATE (Wheels)  │
├───────────────────────────────────┼────────────────────────────────────┤
│ Mechanism: `plan()` Tool Calls    │ Mechanism: Hippocampus KV Slabs    │
│ Target: High-Level Macro Intent   │ Target: Detailed Code / Syntax     │
│ Why: Resolves ambiguity, clarifies│ Why: 100% loss-free, instantaneous │
│      strategy, anchors attention  │      recall without token bloat    │
└───────────────────────────────────┴────────────────────────────────────┘
```

```
               COGNITIVE SATURATION RECOVERY FLOW
               
 [ Engine Detects Saturated Buffer: G < 0.35 or Sp10 > Θ ]
                            │
                            ▼
 [ Step 1: Orchestrator Consolidation Nudge ]
 💭 "[Cognitive load limit reached. Externalizing plan into tool.]"
 🤖 `<|tool_call>call:plan{steps: ["1. Parse AST", "2. Build IR", ...]}`
                            │
                            ▼
 [ Step 2: Hippocampus Subconscious Flush ]
 💾 Flushes active code/derivation (K, V) slabs into `.episodic.mem`.
 🧹 Compacts working buffer, retaining newly structured `plan()` anchor.
                            │
                            ▼
 [ Step 3: Zero-Cost Associative Rehydration ]
 ⚡ When model later executes Step 2, `primeTier3` associative recall
    effortlessly re-injects the exact AST (K, V) slab into L16..L47.
```

This dual-engine approach gives the system **human-level strategic clarity through symbolic articulation** combined with **machine-level eidetic precision through latent memory persistence**.

---

## 7. Hardware Dynamics: Sub-Millisecond Defragmentation on UMA

In discrete GPU systems, memory compaction across PCIe is a bottleneck. However, on unified memory architectures (e.g. AMD Ryzen AI Max+ 395 with LPDDR5X-8000 at $\approx 160\text{ GB/s}$ practical bandwidth), defragmenting the KV cache is nearly instantaneous.

### Memory Arithmetic (Gemma 4 12B)
- **Physical KV Footprint**: 48 Layers $\times$ 2 ($K, V$) $\times$ 8 KV heads $\times$ 128 head dim $\times$ 2 bytes (FP16) = **196.6 KB per slot across the entire model**.
- **Compacting 1,024 Slots**:
  $$\text{Memory Shuffled} = 1024 \times 196.6\text{ KB} \approx 201\text{ MB}$$
- **Compaction Latency**:
  $$\text{Latency} = \frac{201\text{ MB}}{150\text{ GB/s}} \approx 1.34\text{ ms}$$

At $28.5\text{ tok/s}$ ($\approx 35\text{ ms/token}$), a defragmentation pass that executes once every 500–1,000 tokens introduces **less than 0.003 ms per token** of amortized overhead.

### RoPE Rotational Invariance
When slots are evicted non-contiguously and compacted into contiguous physical memory, rotary positional embeddings (RoPE) remain mathematically exact:
1. Rotary angles are embedded directly into key vectors $K$ at logical generation clock $C_i$.
2. When slot $k$ (clock $C_k$) is compacted to a new physical index, its original logical clock $C_k$ is preserved in `token_clocks[slot]`.
3. When query $Q_T$ attends to slot $k$, the dot product evaluates the true relative distance $(T - C_k)$ regardless of where the slot physically resides in RAM.

---

## 8. Phased Implementation Roadmap

```
┌────────────────────────────────────────────────────────────────────────┐
│                   STREAMING TRANSDUCTION ROADMAP                       │
├───────────────────────────────────┬────────────────────────────────────┤
│ PHASE 1: Zero-Tuning Foundation   │ PHASE 2: Native Pacing Fine-Tuning │
│ (Current Open Weights)            │ (Streaming-Tuned Gemma 4)          │
│                                   │                                    │
│ - Attention-Mass Tracking Kernel  │ - Native <|yield|> Special Token   │
│ - UMA KV Compaction Pass (1.3ms)  │ - Micro-Burst Dialogue Dataset     │
│ - Stack-Aware Syntax Tracking     │ - Self-Determined Syntactic Pacing │
│ - Elastic Syntactic Unit Gating   │ - Zero-Heuristic Stream Control    │
│ - Barge-in & Sensory Ingestion    │ - Sub-10ms Barge-In Latency        │
└───────────────────────────────────┴────────────────────────────────────┘
```

### Phase 1: Zero Fine-Tuning Dependency
- **Engine-Level Attention Tracking**: Accumulate attention scores directly in `shaders/decode_attn.wgsl`.
- **Elastic Syntactic Units**: Use stack-depth tracking (tracking unclosed quotes, parentheses, and code fences) combined with local entropy drops to yield cleanly at natural boundaries ($\approx 10\text{--}48$ tokens).
- **Silent Continuation / Auto-Steering**: When an elastic burst yields, the orchestrator inspects external queues (user input, terminal diffs). If quiet, it resumes generation with zero prompt overhead.

### Phase 2: Native Flow Control via Fine-Tuning
- **Native `<|yield|>` Emission**: Fine-tune Gemma 4 on conversational and technical micro-burst datasets so the model learns to emit `<|yield|>` at self-chosen, structurally sound pause points.
- **Bidirectional Transduction**: Eliminates all external heuristic boundary detection, enabling perfectly synchronized, interactive dialogue and multi-step tool execution at line rate.
