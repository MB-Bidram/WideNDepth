# WideNDepth (WND)

Hello, and welcome!

**WideNDepth (WND)** is an experimental neural architecture built around one simple question:

> What if a model could keep a lot of information on hand, but do its step-by-step thinking in a much smaller space?

We're exploring that question in the open, on ordinary hardware (a laptop GPU with 4 GB of memory), and we'll always tell you what we've measured, what we haven't, and what we got wrong along the way.

WND is active research. It is **not** presented as a replacement for Transformers, Mamba, or anything else. It's one idea about how to organize a model's capacity, computation, and memory, and we're finding out how far it goes.

---

## The idea in one picture

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

Here is the same thing in plain words:

1. **Wide Layer**: a roomy place to hold rich information about the input.
2. **Encoder**: turns that information into a useful representation.
3. **Feature Bank**: keeps that richer information around, so nothing important is lost.
4. **Compressor**: squeezes the representation into a small, tidy state.
5. **Depth Layer**: thinks in that small state, over and over (the "xN iterations"). Whenever it needs more detail, it asks the Feature Bank for it through attention.
6. **Output**: turns the final state into the result you want, such as a prediction or generated text.

The Depth Layer does its repeated work in a small space and only reaches for the big storehouse when it needs to.

---

## Why we're building it

In most models, one big stack of layers does everything at once: it stores what the model has learned, processes the current input, and carries out multi-step computation.

```text
Input -> [Layer] -> [Layer] -> [Layer] -> [Layer] -> Output
```

WND tries a different split:

```text
Wide Layer + Feature Bank  ->  rich storage and retrieval
Small Depth Layer          ->  repeated, iterative computation
```

To be honest about it: this doesn't mean knowledge and reasoning live in completely separate places. What a model learns is still spread across all of its weights. A fairer way to say it is:

> WND separates high-capacity feature storage and retrieval from repeated iterative computation.

What we hope this can offer:

- a smaller state during repeated computation,
- access to richer information only when it's needed,
- flexible trade-offs between width, Feature Bank size, and iteration count,
- experiments that are practical on consumer GPUs.

Whether it delivers on all of that is what we're testing.

---

## News and reports

### September 2026: our first deep profile on a 4 GB laptop GPU

We profiled a full training step, kernel by kernel, and traced where the time and memory really go. The findings surprised us, and they're useful.

**The setup**

| | |
|---|---|
| GPU | NVIDIA RTX 3050 Laptop, 4 GB |
| Software | Windows, Python 3.12, PyTorch 2.10, CUDA 12.6, `torch.compile`, bf16 |
| Model | about 42M parameters, `dim=128`, PKM-based Wide layer |
| Blocks | 2 encoder, 2 depth (1 iteration), 1 decoder; 8 state tokens, 16 Feature Bank tokens |
| Data | Wikipedia text, 256 source + 256 target tokens, SmolLM2 tokenizer (about 49k tokens) |

**What we found**

1. **The WND core is not what's slow. The output vocabulary is.** About **90% of GPU time** in a training step went to the final vocabulary projection and the loss over roughly 49,000 tokens. Everything else (Wide layer, Encoder, Compressor, Feature Bank, Depth, Decoder blocks) together took **under 10%**. Inside that remainder, the PKM kernels were the largest piece, at roughly 6% of the step.
2. **Memory ran out for a very specific reason.** At batch size 64, the full logits tensors created four buffers of about 1.5 GiB each, pushing peak memory to **6.35 GiB on a 4 GiB card**. Windows quietly spills the excess to system RAM, and the backward pass slowed to a crawl: a GEMM that ran at about 16 TFLOP/s inside GPU memory ran at about 1 TFLOP/s once its tensors were spilled.
3. **A vocabulary-size detail cost us about 8% of the step.** Our vocabulary (49,154) isn't a multiple of 8, so the compiler padded the logits gradient to 49,160, an extra full pass over the data plus one more 1.5 GiB buffer.
4. **Smaller batches fit comfortably.** At batch size 16 the run needed about 2.8 GiB and ran without spilling. A forward and backward pass took roughly 0.24 s.

These profile numbers cover forward and backward only, without an optimizer step. They also come from profiled runs, which are slower than normal runs, so we read them mainly as shares and ratios, not as absolute speeds.

**What this means**

At this scale, memory and time on a small GPU are dominated by the size of the output vocabulary and how the loss is computed, not by the Wide/Depth design itself. That's good news for the architecture and a useful lesson for anyone training small models with large tokenizers.

**What's next**

- Compute the loss in small chunks so the full logits tensor never has to exist.
- Round the vocabulary up to a multiple of 64.
- Re-profile, and then look closely at the PKM (the next-largest cost) to make the Wide layer cheaper.

### A smaller model, measured earlier

While tuning the runtime, we measured a smaller offline configuration (about 25M parameters, tiny 256-token vocabulary, `dim=256`, PKM Wide with 9,472 memory slots) on the same GPU:

| Measurement | Result |
|---|---:|
| Total parameters | 25,011,842 |
| Parameters in PKM keys and values | about 20.6M (roughly 83%) |
| Inference latency, eager (batch 1, length 32) | 7.73 ms |
| Inference latency, `torch.compile` (reduce-overhead) | 1.32 ms (about 5.9× faster) |
| Eager training step (forward, backward, AdamW) | 37.8 ms median |
| Peak CUDA memory in that training step | about 531 MB allocated, 629 MB reserved |

Two things we liked seeing: most parameters live in the Wide memory, which is the design working as intended, and the main kernel-level cost was PKM's top-k selection over its memory slots, which is exactly the part we plan to make cheaper.

### About our earlier numbers

Our very first preliminary results (a much smaller early prototype) have been retired and are no longer shown. We'd rather show fewer numbers that we trust than a longer list we're not sure about. Fresh quality results, such as validation loss under the current setup, will be added once they're measured properly.

---

## What we know, and what we don't yet

**What we can say today**

- WND builds, trains, compiles, and generates on a 4 GB laptop GPU.
- Under the current setup, the architecture's own compute is a small share of a training step.
- The big costs we found are practical ones (vocabulary size and loss memory) and have clear fixes.

**What isn't proven yet**

WND has not shown general superiority over Transformers, Mamba, state-space models, or other architectures. To find out, we still need:

- matched parameter-count comparisons,
- matched compute and training-token budgets,
- carefully optimized Transformer and Mamba baselines,
- several random seeds, with variance reported,
- ablations for the Feature Bank, retrieval, and Depth Layer,
- more datasets and tasks, and larger-scale runs,
- honest memory reporting, including both allocated and reserved CUDA memory.

We'll fill these in as we go, and we'll say so plainly when a result doesn't hold up.

---

## Reading our numbers fairly

Any result should be read together with its full setup: model configuration, batch size, sequence length, precision, optimizer, tokenizer, PyTorch and CUDA versions, and how memory and time were measured. We try to include these every time, and we'd encourage you to expect the same from anyone else's numbers, too.

---

## Where WND stands

> An experimental architecture that separates rich feature storage and retrieval from iterative computation, being tested openly on consumer hardware.

Thanks for reading, and for being curious about it. 

---

## License

The paper, documentation, diagrams, figures, benchmark results, and other
non-code research materials in this repository are licensed under the
[Creative Commons Attribution 4.0 International License (CC BY 4.0)](
https://creativecommons.org/licenses/by/4.0/).

Copyright © 2026 Mohammadbagher Bidram.

The complete WND implementation, training system, and WiND native runtime are
not included in this repository and remain proprietary.
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
