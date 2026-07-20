# What is WideNDepth (WND)?

**WideNDepth (WND)** is an experimental neural architecture built around one main idea: separating rich information storage from iterative reasoning.

Instead of passing the full, wide representation through every step of computation, WND keeps that richer information available in a **Feature Bank**. It then compresses the information into a smaller state for the **Depth Layer** to process repeatedly.

The Depth Layer can retrieve relevant information from the Feature Bank through attention whenever it needs more context.

In simple terms:

> WND keeps rich information available, while allowing iterative computation to happen in a smaller and more efficient state.

WND is still active research. It is not presented as a proven replacement for Transformers, Mamba, or other architectures. The goal is to explore a different way to organize model capacity, computation, and memory use.

---

# WND's Diagram

```text
Input
  |
  v
Wide Layer
  |
  v
Encoder -------------------------------> Feature Bank
  |                                       |
  v                                       |
Compressor                                |
  |                                       |
  v                                       |
Depth Layer (xN iterations) <--- Attention-based retrieval
  |
  v
Output
```

To explain it simply:

1. The **Wide Layer** is meant to hold richer information or knowledge from the input.

2. The **Encoder** processes that information into a useful representation.

3. The **Feature Bank** stores the original wide representation so that important information remains available later.

4. The **Compressor** reduces the representation into a smaller state.

5. The **Depth Layer** performs iterative reasoning over this compressed state. Whenever it needs additional information, it can retrieve relevant features from the Feature Bank through attention.

6. After the final iteration, the processed state is passed to the **Output** layer for the task, such as prediction, classification, or token generation.

The main purpose of this structure is to let the model work with a smaller state during repeated computation, without losing access to the richer information created earlier.

---

# Why WND?

In many current neural architectures, the same stack of layers is expected to do several things at once:

- store learned patterns and information,
- process the current input,
- maintain useful context,
- and perform multi-step reasoning-like computation.

For example, a simplified deep model may look like this:

```text
Input -> [Layer] -> [Layer] -> [Layer] -> [Layer] -> Output
```

Each layer contributes both learned information and computation. This is effective, but it can also mean that the model repeatedly processes large representations across many layers.

WND explores a different structure:

```text
Input -> Wide representation -> Feature Bank
                                  ^
                                  | retrieval
Compressed state -> Iterative Depth computation -> Output
```

The Wide Layer and Feature Bank are intended to preserve richer information. The Depth Layer is intended to repeatedly work on a smaller compressed state and retrieve richer information only when it is needed.

This is the intuition behind separating **knowledge** from **reasoning**:

```text
Traditional model, simplified:

[ Large Layer Stack ]
     |
     +-> Stores learned information
     |
     +-> Performs computation
```

```text
WND, simplified:

[ Wide Layer + Feature Bank ] -> Rich information storage and retrieval
[ Smaller Depth Layer ]       -> Repeated iterative computation
```

This does not mean knowledge and reasoning are completely separate inside a neural network. Learned information is still distributed across the model's weights, including the Wide Layer, Encoder, Feature Bank, attention mechanism, and Depth Layer.

A more accurate statement is:

> WND separates high-capacity feature storage and retrieval from repeated iterative computation.

Potential benefits being investigated include:

- a smaller state during repeated computation,
- access to richer information through retrieval,
- lower memory use in some configurations,
- flexible tradeoffs between width, Feature Bank size, and iteration count,
- more practical training and experimentation on consumer GPUs.

---

# Our Knowledge of WND

WND is active experimental research. Early results are encouraging, but the architecture is still being tested and validated.

## Current early results

One tested WND configuration had approximately:

| Property | Preliminary result |
|---|---:|
| Model size | ~16 million parameters |
| GPU | NVIDIA RTX 3050 Mobile |
| Dedicated VRAM | 4 GB |
| Observed peak CUDA memory | ~700 MB |
| Training step time | ~100 ms per step |
| Wide dimension | 768 |
| Depth dimension | 256 |
| Feature Bank slots | 32 |
| Depth iterations | 12 |
| Best Wikipedia validation loss | 4.8947 |

These results should always be interpreted together with the full training configuration, including batch size, sequence length, precision, optimizer, tokenizer, PyTorch version, CUDA version, and the exact memory measurement method.

## Early observations

Current experiments suggest that:

- The Feature Bank is important for WND's behavior.
- Removing the Feature Bank substantially reduces performance in tested configurations.
- Removing or weakening attention-based retrieval also harms performance.
- WND has shown promising early behavior on graph-reasoning experiments.
- WND can use a wide representation while keeping its iterative Depth state much smaller.

## What is not proven yet

WND has not yet proven general superiority over Transformers, Mamba, state-space models, or other architectures.

More work is still needed, including:

- matched parameter-count comparisons,
- matched compute and training-token budgets,
- optimized Transformer and Mamba baselines,
- multiple random seeds and variance reporting,
- broader Feature Bank and Depth Layer ablations,
- evaluation across more datasets and tasks,
- larger-scale training experiments,
- careful memory reporting, including allocated and reserved CUDA memory.

## Current research position

WND can currently be described as:

> An experimental neural architecture that separates rich feature storage and retrieval from iterative computation, with encouraging early results for low-memory training on consumer hardware.
