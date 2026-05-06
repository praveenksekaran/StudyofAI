# How Transformers work ?
https://www.youtube.com/watch?v=wjZofJX0v4M
---
# Transformer Architecture: Complete Interview Study Guide

> **Goal:** Understand the Transformer architecture from first principles — well enough to explain, derive, and defend every design decision in a technical interview.

---

## Table of Contents

1. [Why Transformers? The Problem with RNNs](#1-why-transformers-the-problem-with-rnns)
2. [Bird's-Eye View of the Transformer](#2-birds-eye-view-of-the-transformer)
3. [Input Pipeline: Tokenization & Embeddings](#3-input-pipeline-tokenization--embeddings)
4. [Positional Encoding](#4-positional-encoding)
5. [The Encoder](#5-the-encoder)
6. [Self-Attention (Scaled Dot-Product Attention)](#6-self-attention-scaled-dot-product-attention)
7. [Multi-Head Attention](#7-multi-head-attention)
8. [Position-wise Feed-Forward Network](#8-position-wise-feed-forward-network)
9. [Residual Connections & Layer Normalization](#9-residual-connections--layer-normalization)
10. [The Decoder](#10-the-decoder)
11. [Masked Self-Attention](#11-masked-self-attention)
12. [Cross-Attention (Encoder-Decoder Attention)](#12-cross-attention-encoder-decoder-attention)
13. [The Output Layer](#13-the-output-layer)
14. [Full Data Flow: End-to-End](#14-full-data-flow-end-to-end)
15. [Training the Transformer](#15-training-the-transformer)
16. [Variants: Encoder-only, Decoder-only, Encoder-Decoder](#16-variants-encoder-only-decoder-only-encoder-decoder)
17. [Computational Complexity & Efficiency](#17-computational-complexity--efficiency)
18. [Key Design Decisions & Why](#18-key-design-decisions--why)
19. [Common Interview Questions & Answers](#19-common-interview-questions--answers)
20. [Further Reading & Resources](#20-further-reading--resources)

---

## 1. Why Transformers? The Problem with RNNs

Before Transformers, sequence modeling used **Recurrent Neural Networks (RNNs)** and **LSTMs**. Understanding their limitations explains every design decision in the Transformer.

### How RNNs Work

```
         h1        h2        h3        h4
         ↑         ↑         ↑         ↑
"The" → [RNN] → "cat" → [RNN] → "sat" → [RNN] → "on" → [RNN]
         ↓         ↓         ↓         ↓
        out1      out2      out3      out4
```

The hidden state `h_t` carries information from all previous tokens — a **fixed-size bottleneck**.

### Problems with RNNs

```
┌─────────────────────────────────────────────────────────────┐
│  Problem 1: SEQUENTIAL processing — cannot parallelise       │
│                                                              │
│  Step 1 → Step 2 → Step 3 → ... → Step n                   │
│  Must wait for h_{t-1} to compute h_t                       │
│  Training on long sequences = SLOW                           │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Problem 2: VANISHING GRADIENTS over long sequences          │
│                                                              │
│  "The trophy didn't fit in the bag because [it] was too big"│
│                                                              │
│  By the time gradient reaches "trophy", it has been         │
│  multiplied many times → vanishes or explodes               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Problem 3: LONG-RANGE DEPENDENCY — hard to remember         │
│                                                              │
│  "The trophy … 12 words … [it]"                             │
│  The distant token "trophy" has weak influence               │
└─────────────────────────────────────────────────────────────┘
```

### The Transformer Solution

| Problem | RNN | Transformer |
|---|---|---|
| Parallelism | Sequential only | Fully parallel (matrix ops) |
| Long-range deps | Degrades with distance | Constant-distance attention |
| Gradient flow | Vanishes over steps | Direct paths via residuals |
| Training speed | Slow | Fast on modern hardware (GPUs/TPUs) |

> 📖 **Read more:** [Understanding LSTM Networks – Colah's Blog](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

---

## 2. Bird's-Eye View of the Transformer

The original Transformer (Vaswani et al., 2017) was designed for **sequence-to-sequence** tasks like machine translation (`English → French`).

### Full Architecture Diagram

```
                    ENCODER                          DECODER
         ┌─────────────────────────┐    ┌──────────────────────────────┐
         │                         │    │                              │
Input    │  ┌───────────────────┐  │    │  ┌────────────────────────┐ │  Output
Tokens   │  │  Input Embedding  │  │    │  │  Output Embedding      │ │  Tokens
─────────┼─►│  + Pos Encoding   │  │    │  │  + Pos Encoding        │ │◄────────
         │  └────────┬──────────┘  │    │  └──────────┬─────────────┘ │  (shifted
         │           │             │    │             │               │   right)
         │  ┌────────▼──────────┐  │    │  ┌──────────▼─────────────┐ │
         │  │  Encoder Block ×N │  │    │  │  Masked Self-Attention  │ │
         │  │                   │  │    │  │  (sub-layer 1)          │ │
         │  │ ┌───────────────┐ │  │    │  └──────────┬─────────────┘ │
         │  │ │ Self-Attention│ │  │    │             │               │
         │  │ ├───────────────┤ │  │    │  ┌──────────▼─────────────┐ │
         │  │ │     FFN       │ │  │    │  │  Cross-Attention        │◄──── Encoder
         │  │ └───────────────┘ │  │    │  │  (sub-layer 2)         │ │    Output
         │  │   (repeat N=6)    │  │    │  └──────────┬─────────────┘ │
         │  └────────┬──────────┘  │    │             │               │
         │           │             │    │  ┌──────────▼─────────────┐ │
         │           │             │    │  │     FFN                │ │
         │           │             │    │  │  (sub-layer 3)         │ │
         └───────────┼─────────────┘    │  │   (repeat N=6)         │ │
                     │                  │  └──────────┬─────────────┘ │
                     └──────────────────┘             │               │
                                         │  ┌──────────▼─────────────┐ │
                                         │  │  Linear + Softmax      │ │──► Next
                                         │  │  (Output Probabilities) │ │   Token
                                         │  └────────────────────────┘ │
                                         └──────────────────────────────┘
```

### Key Numbers (Base Model from Paper)

| Hyperparameter | Value |
|---|---|
| N (encoder layers) | 6 |
| N (decoder layers) | 6 |
| d_model (embedding dim) | 512 |
| d_ff (FFN hidden dim) | 2048 |
| h (attention heads) | 8 |
| d_k = d_v = d_model/h | 64 |
| Dropout | 0.1 |
| Parameters | ~65M |

> 📖 **Read more:** [Attention Is All You Need (original paper)](https://arxiv.org/abs/1706.03762) | [The Illustrated Transformer – Jay Alammar](https://jalammar.github.io/illustrated-transformer/)

---

## 3. Input Pipeline: Tokenization & Embeddings

Before tensors enter the Transformer, raw text must be converted to numerical vectors.

### Step 1: Tokenization

Text is split into **tokens** using sub-word algorithms like **Byte Pair Encoding (BPE)** or **WordPiece**.

```
Input:  "Transformers are powerful"
Tokens: ["Transform", "##ers", "are", "powerful"]
IDs:    [  8765,        2121,   2024,    3928    ]
```

### Step 2: Token Embedding

Each token ID is looked up in a learned **embedding matrix** `E ∈ ℝ^(V × d_model)`:

```
Vocab size V = 30,000
d_model = 512

token_id = 8765  →  E[8765]  →  vector of shape (512,)

The full sequence of n tokens becomes a matrix X ∈ ℝ^(n × d_model)
```

### Step 3: Scaling

In the paper, embeddings are multiplied by `√d_model` before adding positional encoding:

```
X = E[tokens] × √512

Reason: keeps embedding magnitudes comparable to positional encodings
        (positional encodings have values in [-1, 1])
```

> 📖 **Read more:** [Byte Pair Encoding – Sennrich et al.](https://arxiv.org/abs/1508.07909)

---

## 4. Positional Encoding

Transformers process all tokens **simultaneously** — unlike RNNs, they have no inherent notion of order. Positional encodings inject sequence position information.

### Sinusoidal Positional Encoding (Original Paper)

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Where:
  pos = position in sequence (0, 1, 2, ...)
  i   = dimension index (0, 1, ..., d_model/2)
```

### Visual Pattern

```
  position
  ↑
  │  Each row = positional encoding vector for that position
4 │  [■ □ ■ □ ■ □ ■ □ ■ □ ■ □ ■ □ ■ □]
3 │  [■ □ ■ □ ■ □ ■ □ □ ■ □ ■ □ ■ □ ■]
2 │  [■ □ ■ □ ■ □ □ ■ □ ■ ■ □ ■ □ ■ □]
1 │  [■ □ ■ □ ■ □ ■ □ ■ □ □ ■ □ ■ □ ■]
0 │  [□ □ □ □ □ □ □ □ □ □ □ □ □ □ □ □]
  └──────────────────────────────────►
                                   dimension

  Low dims: high frequency (fine-grained position)
  High dims: low frequency (coarse position)
```

### Why Sinusoidal?

```
Property: PE(pos + k) can be expressed as a linear function of PE(pos)
→ Model can learn to attend by RELATIVE position
→ Generalises to sequence lengths unseen during training
```

### Final Input to Transformer

```
X_final = TokenEmbedding(tokens) × √d_model + PositionalEncoding

Shape: (sequence_length × d_model)
```

### Comparison of Positional Encoding Schemes

| Scheme | Used In | Pros | Cons |
|---|---|---|---|
| Sinusoidal (fixed) | Original Transformer | No learned params; extrapolates | Less flexible |
| Learned Absolute | BERT, GPT-2 | Flexible; adapts to data | Fixed max length |
| Rotary (RoPE) | LLaMA, GPT-NeoX | Great length generalisation | More complex |
| ALiBi | MPT, BLOOM | Simple; no params | May hurt some tasks |

> 📖 **Read more:** [RoPE Paper](https://arxiv.org/abs/2104.09864) | [ALiBi Paper](https://arxiv.org/abs/2108.12409)

---

## 5. The Encoder

The encoder transforms the input sequence into a **rich contextual representation**. It consists of **N=6 identical stacked blocks**.

### Single Encoder Block

```
┌─────────────────────────────────────────────────┐
│                  Encoder Block                   │
│                                                  │
│  Input: X  ────────────────────────────┐        │
│              │                          │        │
│              ▼                          │        │
│         Layer Norm                      │        │
│              │                          │ (skip  │
│              ▼                          │  conn) │
│    Multi-Head Self-Attention            │        │
│              │                          │        │
│              └──────── Add ────────────┘        │
│                          │                       │
│                  ┌───────┘                       │
│                  │                     │         │
│                  ▼                     │         │
│             Layer Norm                 │  (skip  │
│                  │                     │   conn) │
│                  ▼                     │         │
│      Feed-Forward Network (FFN)        │         │
│                  │                     │         │
│                  └──────── Add ────────┘         │
│                          │                       │
│                    Output X'                      │
└─────────────────────────────────────────────────┘
```

### Stack of N Encoders

```
Input Embeddings + Positional Encoding
         │
         ▼
  ┌─────────────┐
  │  Encoder 1  │
  └──────┬──────┘
         │
  ┌──────▼──────┐
  │  Encoder 2  │
  └──────┬──────┘
         │
        ...
         │
  ┌──────▼──────┐
  │  Encoder N  │  ← outputs "encoder memory" — passed to all decoder cross-attn layers
  └──────┬──────┘
         │
  Contextual representations: Z ∈ ℝ^(n × d_model)
```

Each block enriches the representation — later layers capture more abstract semantics.

---

## 6. Self-Attention (Scaled Dot-Product Attention)

Self-attention is the **core operation** of the Transformer. It allows each token to gather information from all other tokens in the sequence.

### Intuition

```
Sentence: "The animal didn't cross the street because it was too tired"

When computing the representation for "it":
  → HIGH attention to "animal"   (it = the animal)
  → LOW  attention to "street"
  → MED  attention to "tired"    (reason for staying)

Self-attention dynamically routes information where it's needed.
```

### Query, Key, Value Projections

For each token vector `x`, we project it into three spaces:

```
Q = X · W_Q    "Query"  — What am I looking for?
K = X · W_K    "Key"    — What do I contain / advertise?
V = X · W_V    "Value"  — What do I output if selected?

W_Q, W_K, W_V ∈ ℝ^(d_model × d_k)    where d_k = 64 (base model)
X              ∈ ℝ^(n × d_model)

→ Q, K, V each ∈ ℝ^(n × d_k)
```

### The Attention Formula

```
                       Q · K^T
Attention(Q, K, V) = softmax(────────) · V
                         √d_k

Step-by-step:

  1. Score matrix:     S = Q · K^T              shape: (n × n)
                       S[i,j] = how much token i attends to token j

  2. Scale:            S = S / √d_k
                       Prevents extremely small gradients from sharp softmax

  3. (Optional mask):  S[i,j] = -∞ for masked positions

  4. Softmax:          A = softmax(S, dim=-1)    each row sums to 1
                       A[i,j] = attention weight: how much token i draws from token j

  5. Weighted sum:     Output = A · V             shape: (n × d_k)
                       Output[i] = weighted combination of all value vectors
```

### Attention Score Matrix Visualized

```
                Keys →  "The"  "cat"  "sat"  "on"  "mat"
                       ┌──────┬──────┬──────┬──────┬──────┐
Queries  "The"  │      │ 0.05 │ 0.60 │ 0.15 │ 0.05 │ 0.15 │
    ↓    "cat"  │      │ 0.10 │ 0.10 │ 0.55 │ 0.10 │ 0.15 │
         "sat"  │      │ 0.05 │ 0.20 │ 0.10 │ 0.30 │ 0.35 │
         "on"   │      │ 0.10 │ 0.10 │ 0.10 │ 0.05 │ 0.65 │
         "mat"  │      │ 0.15 │ 0.25 │ 0.20 │ 0.25 │ 0.15 │
                       └──────┴──────┴──────┴──────┴──────┘
Each row = attention weights (sums to 1)
"The" attends mostly to "cat" (article → noun relationship)
```

### Why Scale by √d_k?

```
As d_k grows → dot products grow in magnitude
              → softmax becomes very "sharp" (peaky)
              → gradients near-zero for most tokens
              → vanishing gradients

Fix: divide by √d_k to restore unit variance

Proof of intuition:
  If q, k ~ N(0,1) with d_k dimensions:
  q·k = Σ q_i·k_i  →  Var[q·k] = d_k
  After scaling:     Var[(q·k)/√d_k] = 1  ✓
```

> 📖 **Read more:** [Illustrated Self-Attention – Jay Alammar](https://jalammar.github.io/illustrated-transformer/) | [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

---

## 7. Multi-Head Attention

Instead of running attention once, the Transformer runs **h = 8 attention heads in parallel**, each learning to focus on different types of relationships.

### Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Multi-Head Attention                       │
│                                                              │
│  Input X (or Q, K, V for cross-attention)                    │
│        │                                                      │
│        ├──── Linear(W_Q1, W_K1, W_V1) ──► Head 1 ──┐        │
│        │                                             │        │
│        ├──── Linear(W_Q2, W_K2, W_V2) ──► Head 2 ──┤        │
│        │                                             │        │
│        ├──── Linear(W_Q3, W_K3, W_V3) ──► Head 3 ──┤ Concat │──► Linear(W_O) ──► Output
│        │                                             │        │
│        ├──── ...                                     │        │
│        │                                             │        │
│        └──── Linear(W_Qh, W_Kh, W_Vh) ──► Head h ──┘        │
│                                                              │
│  head_i = Attention(Q·W_Qi, K·W_Ki, V·W_Vi)                 │
│  MultiHead(Q,K,V) = Concat(head_1,...,head_h) · W_O          │
└──────────────────────────────────────────────────────────────┘

Dimensions:
  Each head:   d_k = d_v = d_model / h = 512 / 8 = 64
  After concat: h × d_v = 8 × 64 = 512 = d_model  ✓
  W_O ∈ ℝ^(d_model × d_model)
```

### Why Multiple Heads?

Different heads specialise in different relationship types:

```
Head 1: Syntactic dependencies    "cat" ↔ "sat" (subject-verb)
Head 2: Semantic similarity       "big" ↔ "large"
Head 3: Coreference               "it" ↔ "animal"
Head 4: Positional adjacency      neighboring tokens
Head 5: Long-range dependencies   beginning ↔ end of sentence
...

Each head sees the full sequence but through different learned projections
→ Each head extracts different kinds of information simultaneously
```

### Computation Cost

Running 8 small heads costs the **same** as running 1 large head:

```
1 head with d_k = 512:    O(n² × 512)
8 heads with d_k = 64:    8 × O(n² × 64) = O(n² × 512)   ← same!

But 8 heads provide richer, multi-faceted representations.
```

> 📖 **Read more:** [What Does BERT Look At? (Attention Analysis)](https://arxiv.org/abs/1906.04341)

---

## 8. Position-wise Feed-Forward Network

After attention aggregates information **across** positions, the FFN applies a non-linear transformation **within** each position independently.

### Formula

```
FFN(x) = max(0, x · W_1 + b_1) · W_2 + b_2    [ReLU version]

FFN(x) = GELU(x · W_1 + b_1) · W_2 + b_2       [GELU version, used in BERT/GPT]

Dimensions:
  Input:  x ∈ ℝ^d_model = ℝ^512
  W_1:    ℝ^(512 × 2048)    ← expand by factor 4
  W_2:    ℝ^(2048 × 512)    ← project back
  Output: ℝ^512
```

### Visualized

```
x (512) ──► W_1 ──► (2048) ──► ReLU/GELU ──► W_2 ──► (512)
             ↑                                  ↑
        expand 4×                          compress back
```

### Key Properties

```
┌──────────────────────────────────────────────────────────┐
│  1. POSITION-WISE: same W_1, W_2 applied at every        │
│     position independently — like a 1×1 convolution      │
│                                                          │
│  2. NO cross-position interaction: FFN handles           │
│     "what to think" after attention handled "where       │
│     to look"                                             │
│                                                          │
│  3. ACTS AS MEMORY: research shows FFN layers store       │
│     factual knowledge as key-value patterns in weights    │
│     (Geva et al., 2021)                                   │
│                                                          │
│  4. 4× EXPANSION: empirically shown to work well;        │
│     larger expansion → more capacity but more memory      │
└──────────────────────────────────────────────────────────┘
```

> 📖 **Read more:** [Transformer Feed-Forward Layers Are Key-Value Memories (Geva et al.)](https://arxiv.org/abs/2012.14913)

---

## 9. Residual Connections & Layer Normalization

Two critical techniques that make training deep Transformers (12, 24, 96+ layers) stable and effective.

### Residual Connections (Skip Connections)

```
Output = x + SubLayer(x)

Without residual:                With residual:
    x                                x ────────────────┐
    │                                │                  │
    ▼                                ▼                  │
SubLayer(x)                      SubLayer(x)            │
    │                                │                  │
    ▼                                └───── Add ────────►
  output                                   │
                                         output

Gradient highway: ∂Loss/∂x = ∂Loss/∂output × (1 + ∂SubLayer/∂x)
The "1" term ensures gradients always flow directly → no vanishing!
```

### Layer Normalization

Normalizes activations across the **feature dimension** for each token independently:

```
LayerNorm(x) = γ · (x - μ) / (σ + ε) + β

Where:
  μ = mean over d_model features for this token
  σ = std over d_model features for this token
  γ, β = learned scale and shift (shape: d_model)
  ε = small constant for numerical stability (1e-6)
```

### LayerNorm vs BatchNorm

```
BatchNorm: normalizes across the BATCH dimension
           ─ problematic for variable-length sequences
           ─ depends on batch size (unstable with small batches)
           ─ different behavior train vs inference

LayerNorm: normalizes across the FEATURE dimension
           ─ each sample normalized independently
           ─ works for any batch size, including 1
           ─ same behavior train and inference  ✓
```

### Pre-Norm vs Post-Norm

```
Post-Norm (original paper):      Pre-Norm (modern standard):
  x' = LN(x + SubLayer(x))         x' = x + SubLayer(LN(x))

Post-Norm: harder to train         Pre-Norm: more stable,
  without warm-up LR scheduling      better gradient flow
                                      used in GPT-2, LLaMA, etc.
```

> 📖 **Read more:** [Layer Normalization Paper](https://arxiv.org/abs/1607.06450) | [On Layer Normalization in Transformer (Pre-Norm analysis)](https://arxiv.org/abs/2002.04745)

---

## 10. The Decoder

The decoder generates the **output sequence** one token at a time. It also has N=6 stacked blocks, but each block has **three** sub-layers instead of two.

### Single Decoder Block

```
┌─────────────────────────────────────────────────────────────┐
│                      Decoder Block                          │
│                                                             │
│  Input: Y  ────────────────────────────────────────┐       │
│              │                                      │       │
│              ▼                                      │       │
│         Layer Norm                                  │       │
│              │                                      │       │
│              ▼                                      │ skip  │
│   MASKED Multi-Head Self-Attention                  │ conn  │
│   (can only see past output tokens)                 │       │
│              │                                      │       │
│              └──────── Add ─────────────────────────┘       │
│                          │                                   │
│              ┌───────────┘ ──────────────────────────┐      │
│              │                                       │      │
│              ▼                                       │      │
│         Layer Norm                                   │      │
│              │                                       │      │
│              ▼  ← Keys and Values from ENCODER       │ skip │
│    Cross-Attention (Encoder-Decoder Attention)        │ conn │
│    Q = from decoder, K = V = from encoder            │      │
│              │                                       │      │
│              └──────── Add ──────────────────────────┘      │
│                          │                                   │
│              ┌───────────┘ ──────────────────────────┐      │
│              │                                       │      │
│              ▼                                       │      │
│         Layer Norm                                   │      │
│              │                                       │ skip │
│              ▼                                       │ conn │
│        Feed-Forward Network                          │      │
│              │                                       │      │
│              └──────── Add ──────────────────────────┘      │
│                          │                                   │
│                    Output Y'                                  │
└─────────────────────────────────────────────────────────────┘
```

### Three Sub-layers in Decoder

```
Sub-layer 1: MASKED Self-Attention
  → Decoder attends to its own previously generated tokens
  → Mask prevents attending to future output tokens (causality)

Sub-layer 2: CROSS-Attention (Encoder-Decoder)
  → Decoder queries the encoder's output memory
  → This is where the decoder "reads" the source sequence

Sub-layer 3: Feed-Forward Network
  → Same as encoder FFN — per-position non-linear transform
```

---

## 11. Masked Self-Attention

The decoder's first sub-layer uses **causal masking** to prevent the decoder from "cheating" by looking at future target tokens.

### The Causal Mask

```
Target sequence: ["<start>", "Je", "suis", "heureux"]
                    pos 0     pos 1  pos 2    pos 3

Attention matrix (4×4), after masking:

                <start>   Je   suis  heureux
               ┌────────┬─────┬──────┬────────┐
<start>        │   ✓    │  ✗  │  ✗   │   ✗    │
Je             │   ✓    │  ✓  │  ✗   │   ✗    │
suis           │   ✓    │  ✓  │  ✓   │   ✗    │
heureux        │   ✓    │  ✓  │  ✓   │   ✓    │
               └────────┴─────┴──────┴────────┘

✗ positions → set to -∞ before softmax → become 0 after softmax

Result: position i can only attend to positions 0 ... i
```

### Why Masking Enables Parallel Training

```
Without masking: would need to generate token by token (slow)

With masking:
  - During TRAINING: feed entire target sequence at once
  - Masking prevents future leakage at each position simultaneously
  - All positions computed in ONE forward pass → FAST training

During INFERENCE: still generates autoregressively (one token at a time)
                  because the next token doesn't exist yet
```

---

## 12. Cross-Attention (Encoder-Decoder Attention)

Cross-attention connects the encoder and decoder. It lets the decoder **"look at" the source sequence** while generating each output token.

### Key Distinction: Where Q, K, V Come From

```
Self-Attention:                 Cross-Attention:
  Q = from same sequence          Q = from DECODER (current state)
  K = from same sequence          K = from ENCODER output (source)
  V = from same sequence          V = from ENCODER output (source)

"What do I want?" (Q) comes from the decoder.
"What's available?" (K, V) comes from the encoder.
```

### Illustration for Translation

```
Source (English): "The cat sat"
Target (French):  "Le chat"  [generating "s'est"]

When generating "s'est" (sat):
  Q = representation of "Le chat s'est" so far
  K, V = encoder's representations of "The", "cat", "sat"

  Cross-attention weights might look like:
    "The"  → 0.10
    "cat"  → 0.10
    "sat"  → 0.80  ← high attention — "s'est" translates "sat"
```

### Cross-Attention Formula

```
CrossAttention(Q_dec, K_enc, V_enc) = softmax(Q_dec · K_enc^T / √d_k) · V_enc

Same formula as self-attention — only Q, K, V sources differ.
```

---

## 13. The Output Layer

After the final decoder block, a linear layer and softmax produce token probabilities.

### Output Head

```
Decoder output:  h ∈ ℝ^(n × d_model)
                        │
                  Linear Layer
                  W_out ∈ ℝ^(d_model × V)    V = vocabulary size
                        │
                  Logits ∈ ℝ^(n × V)
                        │
                    Softmax
                        │
             Probabilities ∈ ℝ^(n × V)    each row sums to 1
                        │
          Next token = argmax or sample
```

### Weight Tying

```
In the original paper and many modern models:

  W_out = Embedding_matrix^T

The output projection shares weights with the input embedding.
Benefits:
  - Fewer parameters to train
  - Consistent representation: tokens similar in embedding space
    are also similar in output space
  - Empirically improves performance
```

### Decoding Strategies

```
Greedy:      always pick argmax → fast but repetitive

Beam Search: maintain top-k sequences at each step
             balance quality vs diversity (used in translation)

Sampling:    sample from softmax distribution → creative text
  Temperature T:
    T→0:  near-greedy, deterministic
    T=1:  true distribution
    T>1:  more random

Top-k:       sample from top k tokens only
Top-p:       sample from smallest set with cumulative prob ≥ p
             (Nucleus Sampling — common in GPT)
```

> 📖 **Read more:** [The Curious Case of Neural Text Degeneration (top-p sampling)](https://arxiv.org/abs/1904.09751)

---

## 14. Full Data Flow: End-to-End

A complete walkthrough of one forward pass for machine translation.

### Task: Translate "The cat sat" → "Le chat s'est assis"

```
═══════════════════════════════════════════════════════════════
ENCODER SIDE
═══════════════════════════════════════════════════════════════

Input text:  "The cat sat"
                │
          Tokenize + lookup embedding matrix
                │
  X = [emb("The"), emb("cat"), emb("sat")]    shape: (3, 512)
                │
          + Positional Encoding
                │
  X_pos = X + PE[0:3]                         shape: (3, 512)
                │
        ┌───────▼────────┐
        │  Encoder Block  │ × 6
        │  Self-Attn      │  Each token attends to all 3 tokens
        │  FFN            │  Enriches representation
        └───────┬────────┘
                │
  Z = Encoder output                           shape: (3, 512)
  "Contextual memory of the English sentence"

═══════════════════════════════════════════════════════════════
DECODER SIDE (step-by-step generation)
═══════════════════════════════════════════════════════════════

Step 1: Input = [<start>]
  Masked Self-Attn: only attends to <start>
  Cross-Attn: Q from decoder, K,V from Z
  Output → Softmax → "Le"  ✓

Step 2: Input = [<start>, "Le"]
  Masked Self-Attn: attends to <start>, "Le"
  Cross-Attn: Q from decoder state, K,V from Z
  Output → Softmax → "chat"  ✓

Step 3: Input = [<start>, "Le", "chat"]
  ...
  Output → Softmax → "s'est"  ✓

Continue until <end> token is generated.
```

---

## 15. Training the Transformer

### Objective: Teacher Forcing + Cross-Entropy

```
During training we feed the CORRECT target sequence (not model predictions)
into the decoder — this is called "Teacher Forcing".

Loss = -Σ log P(y_t | y_<t, X)
        t

= Cross-entropy between predicted distribution and true next token
  at every decoder position, averaged across the sequence and batch.
```

### Training Setup (Original Paper)

```
Dataset:     WMT 2014 English-German (4.5M sentence pairs)
Optimizer:   Adam  (β1=0.9, β2=0.98, ε=1e-9)
Schedule:    Warmup then decay:
             lr = d_model^(-0.5) × min(step^(-0.5), step × warmup^(-1.5))
             warmup_steps = 4000

Regularization:
  - Dropout (p=0.1) on attention weights and FFN
  - Label smoothing (ε=0.1) — softens targets for better calibration

Hardware:    8 × NVIDIA P100 GPUs
Time:        ~12 hours (base model)
```

### Learning Rate Schedule

```
  ▲
  │        /\
lr│       /  \
  │      /    \____________
  │     /
  │    /
  │___/
  └──────────────────────►
    warmup_steps           steps

Warmup: gradually increase lr to avoid instability in early training
Decay:  reduce lr as training progresses (inverse square root)
```

### Label Smoothing

```
Hard targets (standard):   [0, 0, 1, 0, 0]  (one-hot)
Smooth targets:            [0.025, 0.025, 0.9, 0.025, 0.025]

Benefits:
  - Prevents overconfidence
  - Improves BLEU and perplexity
  - Acts as regularization
  ε = 0.1 means: correct class gets (1 - 0.1), rest share 0.1
```

> 📖 **Read more:** [Adam Optimizer](https://arxiv.org/abs/1412.6980)

---

## 16. Variants: Encoder-only, Decoder-only, Encoder-Decoder

The Transformer family has spawned three major architectural lineages:

### Architecture Comparison

```
╔═══════════════╦══════════════════╦═══════════════════╦══════════════════════╗
║               ║  Encoder-Only    ║  Decoder-Only     ║  Encoder-Decoder     ║
╠═══════════════╬══════════════════╬═══════════════════╬══════════════════════╣
║ Example       ║ BERT, RoBERTa    ║ GPT-2/3/4         ║ T5, BART, original   ║
║               ║ DistilBERT       ║ LLaMA, Mistral    ║ Transformer          ║
╠═══════════════╬══════════════════╬═══════════════════╬══════════════════════╣
║ Attention     ║ Bidirectional    ║ Causal (masked)   ║ Both                 ║
║ Type          ║ (sees all tokens)║ (left-only)       ║                      ║
╠═══════════════╬══════════════════╬═══════════════════╬══════════════════════╣
║ Training      ║ Masked LM (MLM)  ║ Causal LM (CLM)   ║ Seq2Seq              ║
║ Objective     ║ Predict [MASK]   ║ Predict next tok  ║ Source → Target      ║
╠═══════════════╬══════════════════╬═══════════════════╬══════════════════════╣
║ Best For      ║ Classification   ║ Text generation   ║ Translation          ║
║               ║ NER, QA, NLI     ║ Completion, Chat  ║ Summarization        ║
╠═══════════════╬══════════════════╬═══════════════════╬══════════════════════╣
║ Cross-Attn    ║ No               ║ No                ║ Yes                  ║
╚═══════════════╩══════════════════╩═══════════════════╩══════════════════════╝
```

### Visual: Which Tokens Can Attend to Which?

```
Encoder-Only (BERT):          Decoder-Only (GPT):
  Each token sees ALL           Each token sees only PAST
  ✓ ✓ ✓ ✓ ✓                    ✓ ✗ ✗ ✗ ✗
  ✓ ✓ ✓ ✓ ✓                    ✓ ✓ ✗ ✗ ✗
  ✓ ✓ ✓ ✓ ✓                    ✓ ✓ ✓ ✗ ✗
  ✓ ✓ ✓ ✓ ✓                    ✓ ✓ ✓ ✓ ✗
  ✓ ✓ ✓ ✓ ✓                    ✓ ✓ ✓ ✓ ✓

Good for understanding          Good for generation
```

> 📖 **Read more:** [BERT Paper](https://arxiv.org/abs/1810.04805) | [T5 Paper](https://arxiv.org/abs/1910.10683)

---

## 17. Computational Complexity & Efficiency

### Self-Attention Complexity

```
Operation              Time              Memory
──────────────────────────────────────────────
Q, K, V projections    O(n · d²)         O(n · d)
Attention scores QK^T  O(n² · d)         O(n²)       ← bottleneck!
Softmax                O(n²)             O(n²)
Attention output A·V   O(n² · d)         O(n²)

Total:                 O(n² · d)         O(n²)

n = sequence length
d = d_model

The n² term is why long contexts (n = 100k+) are expensive.
```

### Comparison with RNN

```
                  Self-Attention    RNN
─────────────────────────────────────────────
Time per layer    O(n² · d)        O(n · d²)
Sequential ops    O(1)             O(n)        ← can't parallelise
Max path length   O(1)             O(n)        ← constant vs linear!
```

Self-attention has **constant maximum path length** between any two tokens — regardless of distance. This is why it handles long-range dependencies so well.

### Efficient Attention Approaches

```
Standard Attention:    O(n²) memory  — impractical for n > 4096

FlashAttention:        O(n) memory   — reorder ops for cache efficiency
                                       same result, 2-4× faster

Sparse Attention:      O(n√n)        — attend only to subset of tokens

Linear Attention:      O(n)          — kernel approximation of softmax

Sliding Window:        O(n · w)      — attend only within window w
(Longformer/Mistral)
```

> 📖 **Read more:** [FlashAttention Paper](https://arxiv.org/abs/2205.14135) | [Longformer Paper](https://arxiv.org/abs/2004.05150)

---

## 18. Key Design Decisions & Why

### Decision 1: Why Dot-Product Attention (not Additive)?

```
Additive attention:    score(q,k) = v^T · tanh(W_q·q + W_k·k)
Dot-product attention: score(q,k) = q · k^T / √d_k

Dot-product: faster (matrix multiply optimised by hardware)
             more memory-efficient
             same quality at larger d_k (with scaling)
```

### Decision 2: Why d_k = d_model/h (not full d_model)?

```
Each head uses reduced dimension d_k = 512/8 = 64.
If each head used d_k = 512:
  - 8× more compute
  - No benefit (concatenation would over-represent same info)
  - Keeping d_k small + concat = same total capacity, better diversity
```

### Decision 3: Why 4× expansion in FFN?

```
Empirically, d_ff = 4 × d_model consistently outperforms
other expansion ratios in benchmarks.
Hypothesis: larger FFN = more capacity to store knowledge
            as implicit key-value memories.
```

### Decision 4: Why 6 layers (not more/fewer)?

```
Original paper tried 2, 4, 6, 8 layers.
6 gave the best trade-off on WMT translation benchmarks.
Modern models use far more (GPT-3: 96, LLaMA: 32-80).
More layers → richer representations, but diminishing returns.
```

### Decision 5: Why sinusoidal (not learned) positional encoding?

```
Original paper used both, found similar quality.
Sinusoidal chosen because:
  - No extra parameters
  - Extrapolates to unseen sequence lengths
  - Encodes relative distance (key property)
Modern models mostly use learned embeddings anyway.
```

---

## 19. Common Interview Questions & Answers

### Q1: What is the attention mechanism and why is it important?

**A:** Attention allows each token to gather information from all other tokens in the sequence with learned, input-dependent weights. Unlike RNNs, which compress history into a fixed vector, attention provides a direct, dynamic path between any two tokens — crucial for long-range dependencies. It's important because it's what gives Transformers their ability to model complex linguistic relationships efficiently.

### Q2: What is the complexity of self-attention? Why is it a problem?

**A:** `O(n²·d)` time and `O(n²)` memory, where n is sequence length. The quadratic term in n becomes a bottleneck for long sequences (e.g., a 100K-token document needs 10 billion attention scores). Solutions include FlashAttention (memory-efficient), sparse attention, and sliding-window approaches.

### Q3: Why do we use multi-head attention instead of single-head?

**A:** Multiple heads allow the model to simultaneously attend to different types of relationships — syntactic, semantic, coreference, positional — at different representation subspaces. Single-head attention averages everything into one representation, losing this specialisation. The total computation is the same since each head uses `d_model/h` dimensions.

### Q4: Why scale attention scores by √d_k?

**A:** Without scaling, as d_k grows, dot products grow in magnitude, pushing the softmax into regions with near-zero gradients (vanishing gradients). Dividing by `√d_k` restores unit variance: if q and k have unit variance components, their dot product has variance d_k, and dividing by `√d_k` brings it back to variance 1.

### Q5: What is the difference between self-attention and cross-attention?

**A:** In self-attention, Q, K, and V all come from the same sequence — the sequence attends to itself. In cross-attention (encoder-decoder attention), Q comes from the decoder (current state) while K and V come from the encoder output (source sequence). Cross-attention lets the decoder "read" the source sequence at each generation step.

### Q6: What is the role of residual connections in Transformers?

**A:** Residual connections (`x' = x + SubLayer(x)`) create "gradient highways" — during backpropagation, the gradient of the loss with respect to x includes a direct term (the "1" from the derivative of the residual), preventing gradient vanishing through deep stacks. They also allow each layer to incrementally refine the representation rather than completely transforming it.

### Q7: Why is Layer Norm used instead of Batch Norm?

**A:** Batch Norm normalizes across the batch dimension, which is problematic for variable-length sequences and behaves differently during training vs inference. Layer Norm normalizes across the feature dimension for each example independently, working consistently regardless of batch size or sequence length — making it the right choice for NLP.

### Q8: Explain teacher forcing. What is exposure bias?

**A:** During training, the decoder receives the ground-truth previous tokens as input (even if its own prediction was wrong). This is teacher forcing — it stabilises and speeds up training. The downside is **exposure bias**: at inference time, the decoder uses its own (potentially wrong) predictions, creating a distribution shift from training. This can cause error accumulation.

### Q9: What would happen if we removed positional encoding?

**A:** The model would be **permutation invariant** — "dog bites man" and "man bites dog" would produce identical representations. Positional encodings break this symmetry by injecting position information, allowing the model to learn order-dependent patterns like subject-verb agreement and word order in translation.

### Q10: How are Encoder-only, Decoder-only, and Encoder-Decoder models different in application?

**A:** **Encoder-only** (BERT): bidirectional attention → full context → ideal for understanding tasks (classification, NER, QA). **Decoder-only** (GPT): causal attention → generates left-to-right → ideal for generation tasks (chatbots, code completion). **Encoder-Decoder** (T5, BART): encoder builds source representation, decoder generates conditioned on it → ideal for transformation tasks (translation, summarization).

---

## 20. Further Reading & Resources

### Essential Papers

| Paper | Why It Matters |
|---|---|
| [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) | Original Transformer — must read |
| [BERT (2018)](https://arxiv.org/abs/1810.04805) | Encoder-only; bidirectional pre-training |
| [GPT-2 (2019)](https://openai.com/research/better-language-models) | Decoder-only; scaling |
| [T5 (2019)](https://arxiv.org/abs/1910.10683) | Unified text-to-text framework |
| [Chinchilla (2022)](https://arxiv.org/abs/2203.15556) | Optimal scaling laws |
| [FlashAttention (2022)](https://arxiv.org/abs/2205.14135) | Efficient attention in practice |
| [RoPE (2021)](https://arxiv.org/abs/2104.09864) | Rotary positional encoding |

### Interactive Visualizations & Blogs

| Resource | What You'll Learn |
|---|---|
| [The Illustrated Transformer – Jay Alammar](https://jalammar.github.io/illustrated-transformer/) | Best visual walkthrough |
| [The Illustrated BERT – Jay Alammar](https://jalammar.github.io/illustrated-bert/) | Encoder pre-training |
| [LLM Visualization (3D interactive)](https://bbycroft.net/llm) | Step-through a real GPT |
| [Distill.pub – Attention and Augmented RNNs](https://distill.pub/2016/augmented-rnns/) | Attention origins |
| [Lilian Weng – Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) | Deep dive on attention variants |

### Video Lectures

| Resource | What You'll Learn |
|---|---|
| [Andrej Karpathy — Build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) | Code a Transformer live |
| [Stanford CS224N Lecture 9 – Transformers](https://www.youtube.com/watch?v=ptuGllU5SQQ) | Academic depth |
| [3Blue1Brown — Attention in Transformers](https://www.youtube.com/watch?v=eMlx5fFNoYc) | Intuitive visual explanation |

### Code to Study

| Resource | Why |
|---|---|
| [nanoGPT – Karpathy](https://github.com/karpathy/nanoGPT) | Clean 300-line Transformer |
| [Annotated Transformer – Harvard NLP](https://nlp.seas.harvard.edu/annotated-transformer/) | Paper code side-by-side |
| [HuggingFace Transformers source](https://github.com/huggingface/transformers) | Production implementation |

---

## Quick Reference: Transformer Component Summary

```
╔══════════════════════╦═══════════════════════════════════════════════════╗
║ Component            ║ Purpose & Key Formula                             ║
╠══════════════════════╬═══════════════════════════════════════════════════╣
║ Token Embedding      ║ integer ID → dense vector; E ∈ ℝ^(V×d)          ║
║ Positional Encoding  ║ inject order; PE(p,2i) = sin(p/10000^(2i/d))    ║
║ Self-Attention       ║ attend to self; softmax(QK^T/√d_k)V              ║
║ Multi-Head Attention ║ h parallel heads, concat + project               ║
║ Masked Attention     ║ causal mask: -∞ for future positions             ║
║ Cross-Attention      ║ Q←decoder, K,V←encoder; bridge seq2seq          ║
║ FFN                  ║ per-position MLP; expand 4×, activate, compress  ║
║ Residual Connection  ║ x' = x + SubLayer(x); prevents gradient vanish   ║
║ Layer Norm           ║ normalize over features; stable training          ║
║ Output Head          ║ linear d→V + softmax → next token probs          ║
╚══════════════════════╩═══════════════════════════════════════════════════╝
```

---

*Study guide compiled for Transformer Architecture interview preparation.*
*All diagrams are ASCII art for portability. Render in any Markdown viewer.*




---
# GPT Architecture: Complete Interview Study Guide

> **Goal:** Understand GPT (Generative Pre-trained Transformer) from first principles — enough to answer deep technical interview questions with confidence.

---

## Table of Contents

1. [Big Picture: What is GPT?](#1-big-picture-what-is-gpt)
2. [Foundations: The Transformer Architecture](#2-foundations-the-transformer-architecture)
3. [Tokenization](#3-tokenization)
4. [Embeddings](#4-embeddings)
5. [Positional Encoding](#5-positional-encoding)
6. [The Transformer Decoder Block](#6-the-transformer-decoder-block)
7. [Attention Mechanism (Deep Dive)](#7-attention-mechanism-deep-dive)
8. [Multi-Head Attention](#8-multi-head-attention)
9. [Masked Self-Attention (Causal Masking)](#9-masked-self-attention-causal-masking)
10. [Feed-Forward Network (FFN)](#10-feed-forward-network-ffn)
11. [Layer Normalization & Residual Connections](#11-layer-normalization--residual-connections)
12. [The Output Layer: Language Model Head](#12-the-output-layer-language-model-head)
13. [Pre-training Objective](#13-pre-training-objective)
14. [Fine-tuning & RLHF](#14-fine-tuning--rlhf)
15. [GPT-1 → GPT-4: Evolution](#15-gpt-1--gpt-4-evolution)
16. [Key Hyperparameters & Scaling Laws](#16-key-hyperparameters--scaling-laws)
17. [Common Interview Questions & Answers](#17-common-interview-questions--answers)
18. [Further Reading & Resources](#18-further-reading--resources)

---

## 1. Big Picture: What is GPT?

**GPT** = **G**enerative **P**re-trained **T**ransformer.

It is a **language model** — a system trained to predict the next token given a sequence of previous tokens. At inference time, it generates text **autoregressively**: one token at a time, each time feeding the output back as input.

```
Input:  "The cat sat on the"
Output: "mat"  →  feed back  →  "mat ."  →  ...
```

### Three-phase lifecycle of GPT

```
┌───────────────────────────────────────────────────────────────┐
│  PHASE 1: Pre-training                                         │
│  Massive text corpus (internet, books, code)                   │
│  Objective: predict next token (causal language modeling)      │
│  Result: a general-purpose "world model" in weights            │
└───────────────────────────┬───────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────┐
│  PHASE 2: Supervised Fine-Tuning (SFT)                         │
│  Curated (prompt, ideal response) pairs                        │
│  Teach the model to follow instructions                         │
└───────────────────────────┬───────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────┐
│  PHASE 3: RLHF (Reinforcement Learning from Human Feedback)    │
│  Human preferences used to train a Reward Model               │
│  PPO optimizes the policy to maximise reward                   │
└───────────────────────────────────────────────────────────────┘
```

> 📖 **Read more:** [OpenAI GPT-3 paper](https://arxiv.org/abs/2005.14165) | [OpenAI InstructGPT (RLHF)](https://arxiv.org/abs/2203.02155)

---

## 2. Foundations: The Transformer Architecture

GPT is based on the **decoder-only** variant of the original Transformer (Vaswani et al., 2017 — *"Attention Is All You Need"*).

### Original Transformer vs GPT

| Feature | Original Transformer | GPT |
|---|---|---|
| Architecture | Encoder + Decoder | **Decoder only** |
| Task | Seq2Seq (translation) | Language Modeling |
| Attention type | Bidirectional (encoder) + Masked (decoder) | **Masked (causal) only** |
| Cross-attention | Yes | No |

### High-level GPT Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    GPT Model                             │
│                                                          │
│  Input tokens:  [T1]  [T2]  [T3]  ...  [Tn]            │
│                   │     │     │           │              │
│              ┌────▼─────▼─────▼───────────▼────┐        │
│              │     Token Embedding Table         │        │
│              └────────────────┬────────────────┘        │
│                               │                          │
│              ┌────────────────▼────────────────┐        │
│              │    + Positional Encoding          │        │
│              └────────────────┬────────────────┘        │
│                               │                          │
│              ┌────────────────▼────────────────┐        │
│              │   Transformer Decoder Block 1    │        │
│              │   ┌───────────────────────────┐  │        │
│              │   │  Masked Multi-Head Attn   │  │        │
│              │   ├───────────────────────────┤  │        │
│              │   │  Feed-Forward Network     │  │        │
│              │   └───────────────────────────┘  │        │
│              └────────────────┬────────────────┘        │
│                              ...                         │
│              ┌────────────────▼────────────────┐        │
│              │   Transformer Decoder Block N    │        │
│              └────────────────┬────────────────┘        │
│                               │                          │
│              ┌────────────────▼────────────────┐        │
│              │   Layer Norm + Linear (LM Head)  │        │
│              └────────────────┬────────────────┘        │
│                               │                          │
│              ┌────────────────▼────────────────┐        │
│              │   Softmax → Probability over     │        │
│              │   Vocabulary (next token)         │        │
│              └─────────────────────────────────┘        │
└─────────────────────────────────────────────────────────┘
```

> 📖 **Read more:** [Attention Is All You Need (original paper)](https://arxiv.org/abs/1706.03762) | [The Illustrated Transformer – Jay Alammar](https://jalammar.github.io/illustrated-transformer/)

---

## 3. Tokenization

Before text enters the model, it must be converted into **tokens** — discrete integer IDs.

### What is a Token?

A token is roughly a word-piece, sub-word, or character chunk. GPT uses **Byte Pair Encoding (BPE)**.

```
"unhappiness"  →  ["un", "hap", "pi", "ness"]
                       ↓       ↓      ↓      ↓
                      [342]  [6201] [802] [1215]   (integer IDs)
```

### Why Sub-word Tokenization?

- **Avoids huge vocabulary** (one token per word = millions)  
- **Handles unknown words** (OOV) gracefully  
- **Captures morphology** (prefix/suffix patterns)

### BPE Algorithm (simplified)

```
1. Start with character-level vocabulary
2. Count all adjacent pair frequencies
3. Merge the most frequent pair into a new token
4. Repeat until vocabulary size V is reached
```

GPT-2 uses V = 50,257 tokens. GPT-4 uses ~100,000 tokens (cl100k_base).

> 📖 **Read more:** [BPE Paper](https://arxiv.org/abs/1508.07909) | [Tiktokenizer (interactive)](https://tiktokenizer.vercel.app/)

---

## 4. Embeddings

Once tokenized, each integer ID is mapped to a **dense vector** via a learnable embedding matrix.

### Token Embedding

```
Vocabulary size  V = 50,257
Embedding dim    d = 768  (GPT-2 small)

Embedding matrix E ∈ R^(V × d)

token_id = 342  →  E[342]  →  vector of shape (768,)
```

This matrix is **learned during training** — semantically similar tokens end up near each other in the high-dimensional space.

### Embedding Dimension by Model

| Model | d_model | Layers | Heads | Parameters |
|---|---|---|---|---|
| GPT-2 Small | 768 | 12 | 12 | 117M |
| GPT-2 Medium | 1024 | 24 | 16 | 345M |
| GPT-2 Large | 1280 | 36 | 20 | 774M |
| GPT-2 XL | 1600 | 48 | 25 | 1.5B |
| GPT-3 | 12288 | 96 | 96 | 175B |

---

## 5. Positional Encoding

Transformers have **no inherent sense of order** — without positional information, "dog bites man" and "man bites dog" look identical.

### GPT-2: Learned Positional Embeddings

GPT-2 adds a **learned positional embedding** for each position 0 … (context_length - 1).

```
Position matrix  P ∈ R^(max_seq_len × d)

Final input = Token_Embedding + Positional_Embedding
```

This is different from the original Transformer which used **fixed sinusoidal** encodings.

### Comparison of Positional Encoding Schemes

| Scheme | Used In | Key Property |
|---|---|---|
| Sinusoidal (fixed) | Original Transformer | Generalises to unseen lengths |
| Learned absolute | GPT-1, GPT-2 | Simple, works well within training length |
| Rotary (RoPE) | LLaMA, GPT-NeoX | Relative positions via rotation — extrapolates better |
| ALiBi | MPT, BLOOM | Adds bias to attention scores — no learned params |

> 📖 **Read more:** [RoPE Paper](https://arxiv.org/abs/2104.09864) | [ALiBi Paper](https://arxiv.org/abs/2108.12409)

---

## 6. The Transformer Decoder Block

Each **decoder block** (GPT has N of them, stacked) consists of:

```
┌─────────────────────────────────────────────────┐
│              Transformer Decoder Block            │
│                                                   │
│  Input x  ──────────────────────────────┐        │
│             │                           │        │
│             ▼                           │        │
│       Layer Norm 1                      │        │
│             │                           │        │
│             ▼                           │        │
│    Masked Multi-Head                    │ (residual
│    Self-Attention                       │  connection)
│             │                           │        │
│             └──────── + ───────────────┘        │
│                        │                          │
│                        │  ┌──────────────────┐   │
│                        └──┤                  │   │
│                           │  x + Attention   │   │
│                        ┌──┤                  │   │
│                        │  └──────────────────┘   │
│                        │                          │
│             ┌──────────┘                          │
│             │                           │        │
│             ▼                           │        │
│       Layer Norm 2                      │        │
│             │                           │        │
│             ▼                           │        │
│    Feed-Forward Network                 │ (residual)
│    (MLP)                                │        │
│             │                           │        │
│             └──────── + ───────────────┘        │
│                        │                          │
│                   Output x'                       │
└─────────────────────────────────────────────────┘
```

**Two sub-layers per block:**
1. **Masked Multi-Head Self-Attention** — lets each token attend to previous tokens
2. **Feed-Forward Network (MLP)** — applies non-linear transformation per position

Both wrapped with **Layer Norm** (pre-norm in GPT-2+) and **Residual connections**.

---

## 7. Attention Mechanism (Deep Dive)

Attention is the **core innovation** of the Transformer. It allows every token to "look at" other tokens and selectively gather information.

### Scaled Dot-Product Attention

Given a sequence of vectors, we derive three matrices:

```
Q = X · W_Q    (Queries)   — "What am I looking for?"
K = X · W_K    (Keys)      — "What do I contain?"
V = X · W_V    (Values)    — "What do I output if selected?"

Where X ∈ R^(seq_len × d_model)
      W_Q, W_K, W_V ∈ R^(d_model × d_k)
```

### Attention Score Computation

```
                         QK^T
Attention(Q,K,V) = softmax(────) · V
                          √d_k

Step by step:
  1. Compute scores:   S = Q · K^T          shape: (seq, seq)
  2. Scale:            S = S / √d_k          prevents vanishing gradients
  3. Mask:             S[i,j] = -∞ if j > i  causal masking
  4. Softmax:          A = softmax(S)         attention weights sum to 1
  5. Weighted sum:     Output = A · V
```

### Why scale by √d_k?

As `d_k` grows, dot products grow in magnitude → softmax becomes very "peaky" → gradients vanish.  
Dividing by `√d_k` keeps variance ≈ 1.

### Intuitive Example

```
Sentence: "The animal didn't cross the street because it was too tired"

When computing attention for "it":
  - "animal" gets HIGH attention weight  ✓  (it refers to the animal)
  - "street" gets LOW attention weight
  - "tired" gets MEDIUM attention weight
```

> 📖 **Read more:** [Illustrated Attention – Jay Alammar](https://jalammar.github.io/illustrated-transformer/) | [Attention Paper](https://arxiv.org/abs/1706.03762)

---

## 8. Multi-Head Attention

Instead of one attention operation, GPT runs **h parallel attention heads**, each with its own learned projections.

```
┌──────────────────────────────────────────────────────────┐
│                   Multi-Head Attention                    │
│                                                           │
│  Input X                                                  │
│     │                                                     │
│     ├──────────── Head 1 ─────────────┐                  │
│     │   Q1=X·W_Q1, K1=X·W_K1, V1=X·W_V1                │
│     │   head_1 = Attention(Q1,K1,V1)  │                  │
│     │                                  │                  │
│     ├──────────── Head 2 ─────────────┤                  │
│     │   head_2 = Attention(Q2,K2,V2)  │                  │
│     │                                  │  Concat          │
│     ├──────────── ...   ─────────────┤──────────► · W_O │
│     │                                  │                  │
│     └──────────── Head h ─────────────┘                  │
│         head_h = Attention(Qh,Kh,Vh)                     │
│                                                           │
└──────────────────────────────────────────────────────────┘

MultiHead(Q,K,V) = Concat(head_1, ..., head_h) · W_O
```

### Why Multiple Heads?

Each head can specialise in a different **type of relationship**:
- Head 1: syntactic dependencies
- Head 2: semantic similarity
- Head 3: coreference (e.g., pronoun ↔ noun)
- Head 4: positional proximity

**Dimensionality:** Each head operates on `d_k = d_model / h`, so total computation stays the same.

For GPT-2 Small: `d_model=768, h=12` → each head has `d_k=64`

---

## 9. Masked Self-Attention (Causal Masking)

GPT is a **causal** (left-to-right) language model. It must **not** look at future tokens.

### The Causal Mask

```
Attention matrix for sequence ["I", "love", "GPT"]

Without mask:        With causal mask:
  I  love  GPT         I  love  GPT
I  [✓   ✓    ✓ ]   I  [✓   ✗    ✗ ]
love [✓   ✓    ✓ ]   love [✓   ✓    ✗ ]
GPT  [✓   ✓    ✓ ]   GPT  [✓   ✓    ✓ ]

Masked positions → set to -∞ before softmax → become 0 after softmax
```

This ensures:
- Token at position `i` can only attend to positions `0 … i`
- Training can be done in **parallel** (all positions simultaneously)
- Inference is **sequential** (one token at a time)

---

## 10. Feed-Forward Network (FFN)

After attention, each position goes through a **position-wise** Feed-Forward Network independently.

```
FFN(x) = max(0, x·W_1 + b_1) · W_2 + b_2

Or with GELU activation (used in GPT-2+):

FFN(x) = GELU(x·W_1 + b_1) · W_2 + b_2

Dimensions:
  Input:   x ∈ R^d_model
  W_1:     R^(d_model × 4·d_model)   ← expands by 4×
  W_2:     R^(4·d_model × d_model)   ← projects back
```

### Key Points

- **Applied independently** at each position — no information flows between positions here
- **4× expansion** is a standard design choice (empirically works well)
- **GELU** activation (Gaussian Error Linear Unit) is smoother than ReLU:

```
GELU(x) ≈ x · σ(1.702 · x)

         /
        /
_______/          ← smooth, allows negative values
```

> 📖 **Read more:** [GELU Paper](https://arxiv.org/abs/1606.08415)

---

## 11. Layer Normalization & Residual Connections

Two critical stability techniques used throughout GPT.

### Residual Connections

Also called **skip connections** (from ResNets). They add the input directly to the output:

```
x' = x + SubLayer(x)
```

**Why?** Allows gradients to flow directly through the network during backpropagation, mitigating the vanishing gradient problem in deep networks.

### Layer Normalization

Normalizes activations **across the feature dimension** (not the batch):

```
LayerNorm(x) = γ · (x - μ) / (σ + ε) + β

Where:
  μ = mean of x across features
  σ = std of x across features
  γ, β = learned scale and shift parameters
```

**Pre-Norm vs Post-Norm:**

```
Post-Norm (original Transformer):
  x' = LayerNorm(x + SubLayer(x))

Pre-Norm (GPT-2+):
  x' = x + SubLayer(LayerNorm(x))   ← more stable training
```

GPT-2 and later use **Pre-Norm** for more stable gradient flow.

> 📖 **Read more:** [Layer Norm Paper](https://arxiv.org/abs/1607.06450)

---

## 12. The Output Layer: Language Model Head

After N decoder blocks, the final output is projected to vocabulary space.

```
Block output:   h ∈ R^(seq_len × d_model)
                      │
               Layer Norm
                      │
            Linear (W_lm_head ∈ R^(d_model × V))
                      │
                  Logits  ∈ R^(seq_len × V)
                      │
                  Softmax
                      │
              Probability over vocabulary

Next token = argmax(probs)  [greedy]
           or sample(probs) [sampling]
```

### Weight Tying

In most GPT variants, the **LM head weight W_lm_head is tied to the token embedding matrix E**.

This means `W_lm_head = E^T`, reducing parameters and improving training efficiency.

---

## 13. Pre-training Objective

GPT is trained with **Causal Language Modeling (CLM)** — also called the **next-token prediction** objective.

### Loss Function

```
Given tokens [t_1, t_2, ..., t_n]:

L = - (1/n) Σ log P(t_i | t_1, ..., t_{i-1})
             i=1

= Cross-Entropy loss averaged over all positions
```

### Training Loop (simplified)

```
for batch in data:
    tokens = tokenize(batch)                  # (B, T)
    logits = model(tokens[:, :-1])            # predict using all but last
    loss   = cross_entropy(logits, tokens[:, 1:])  # compare with all but first
    loss.backward()
    optimizer.step()
```

Every token in every sequence contributes to the loss — extremely **data efficient** compared to supervised classification tasks.

### What does the model actually learn?

By predicting the next token on massive corpora, the model implicitly learns:
- Grammar and syntax
- Facts and world knowledge
- Reasoning patterns
- Code structure
- Mathematics (to some degree)

> 📖 **Read more:** [GPT-1 Paper (Radford et al., 2018)](https://openai.com/research/language-unsupervised)

---

## 14. Fine-tuning & RLHF

Pre-trained GPT is a powerful "completion" model but not yet an "assistant." Alignment requires further training.

### Stage 1: Supervised Fine-Tuning (SFT)

```
Dataset: { (prompt, ideal_response) } pairs
          curated by human labellers

Training: Same CLM loss, but only on high-quality conversations
Result:   Model learns instruction-following behavior
```

### Stage 2: Reward Model (RM) Training

```
1. Sample multiple completions from SFT model for the same prompt
2. Human rankers order completions best → worst
3. Train a Reward Model to predict human preference scores
   RM: (prompt, completion) → scalar score
```

### Stage 3: PPO (Proximal Policy Optimization)

```
┌─────────────────────────────────────────────────┐
│             RLHF Training Loop                   │
│                                                   │
│  Prompt → SFT Model (policy) → Completion         │
│                                    │              │
│                             Reward Model          │
│                                    │              │
│                             Reward Signal r        │
│                                    │              │
│  KL Penalty: - β · KL(π || π_SFT)  │              │
│  (prevents policy from drifting    │              │
│   too far from SFT model)          │              │
│                                    ▼              │
│                          PPO update policy        │
└─────────────────────────────────────────────────┘

Total reward = r - β · KL(π_θ || π_SFT)
```

> 📖 **Read more:** [InstructGPT Paper](https://arxiv.org/abs/2203.02155) | [RLHF Blog – HuggingFace](https://huggingface.co/blog/rlhf)

---

## 15. GPT-1 → GPT-4: Evolution

### Timeline & Scale

```
Year    Model     Params    Context    Key Innovation
────────────────────────────────────────────────────────────
2018    GPT-1     117M      512        Pre-train + fine-tune
2019    GPT-2     1.5B      1024       Larger scale; zero-shot
2020    GPT-3     175B      2048       Few-shot in-context learning
2022    InstructGPT  1.3B+  2048       RLHF alignment
2023    GPT-3.5   ~175B     4096       ChatGPT; improved RLHF
2023    GPT-4     ~1T?      128K       Multi-modal; much stronger reasoning
```

### Key Milestones

- **GPT-1:** Proved pre-train → fine-tune paradigm works for NLP
- **GPT-2:** Showed emergent zero-shot capability — surprised the field
- **GPT-3:** In-context learning: just give examples in the prompt, no weight update needed
- **InstructGPT:** RLHF makes models helpful, harmless, honest
- **GPT-4:** Multimodal input, dramatically improved reasoning, longer context

> 📖 **Read more:** [GPT-2 Paper](https://openai.com/research/better-language-models) | [GPT-3 Paper](https://arxiv.org/abs/2005.14165) | [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774)

---

## 16. Key Hyperparameters & Scaling Laws

### Core Hyperparameters

| Hyperparameter | Symbol | Description |
|---|---|---|
| Vocabulary size | V | Number of unique tokens |
| Embedding dim | d_model | Vector size throughout model |
| Number of layers | N | Decoder blocks stacked |
| Number of heads | h | Parallel attention heads |
| Head dimension | d_k = d_model/h | Per-head QKV size |
| FFN expansion | 4 × d_model | Hidden size in FFN |
| Context length | T | Max tokens in one forward pass |
| Dropout | p | Regularization (training only) |

### Chinchilla Scaling Laws

The DeepMind **Chinchilla** paper (2022) established:

```
For compute-optimal training:

  N_opt ∝ C^0.5   (model parameters)
  D_opt ∝ C^0.5   (training tokens)

  → ~20 tokens per parameter is optimal

  Example: 70B model → train on ~1.4T tokens
```

This overturned GPT-3 which was *undertrained* (175B params, only 300B tokens).

> 📖 **Read more:** [Chinchilla Paper](https://arxiv.org/abs/2203.15556) | [Neural Scaling Laws (OpenAI)](https://arxiv.org/abs/2001.08361)

---

## 17. Common Interview Questions & Answers

### Q1: Why can't GPT attend to future tokens?

**A:** GPT is trained to predict the next token. If it could see future tokens, it would trivially "cheat" by copying the answer — learning nothing. The causal mask enforces this constraint. During training, we can still compute all positions in parallel because the mask prevents future information from flowing.

### Q2: What is the complexity of self-attention?

**A:** `O(n² · d)` where n = sequence length, d = embedding dimension. The quadratic term in n is why long contexts are expensive. Solutions: sparse attention (Longformer), linear attention approximations (Performers), sliding window attention (Mistral).

### Q3: Why LayerNorm instead of BatchNorm in Transformers?

**A:** BatchNorm normalizes across the batch dimension, which is unstable for variable-length sequences and small batches. LayerNorm normalizes across the feature dimension for each example independently — more robust for sequence data.

### Q4: What is the role of the FFN in a Transformer?

**A:** The attention mechanism performs weighted averaging — it routes and mixes information between positions, but is linear in values. The FFN applies a non-linear transformation *per position*, adding expressive power. Research suggests the FFN acts as a "key-value memory" storing factual knowledge.

### Q5: What is in-context learning? Does it update weights?

**A:** In-context learning is when the model adapts its behavior based on examples provided in the prompt alone, **without any gradient update or weight change**. The model uses its attention mechanism to "learn" the pattern from the examples. The weights remain frozen.

### Q6: What is temperature in sampling? How does it affect output?

**A:**
```
logits_scaled = logits / temperature

Temperature → 0:  near-deterministic (greedy), repetitive
Temperature = 1:  sample from true distribution
Temperature > 1:  more random, creative, incoherent
```

### Q7: What is the difference between GPT (decoder-only) and BERT (encoder-only)?

| | GPT | BERT |
|---|---|---|
| Architecture | Decoder-only | Encoder-only |
| Attention | Causal (left-only) | Bidirectional |
| Training | Next-token prediction | Masked LM + NSP |
| Use case | Generation | Classification, QA, embedding |

### Q8: What is KV Cache and why does it matter?

**A:** During autoregressive generation, the Keys and Values for all past tokens are recomputed every step (expensive). KV Cache stores these computed K and V tensors from previous steps, so only the **new token's** K,V need to be computed each step. This makes inference `O(n)` per step instead of `O(n²)`.

### Q9: What are positional encodings and why are they needed?

**A:** Transformers process all tokens in parallel through matrix multiplications — there is no inherent ordering. Positional encodings inject position information by adding a position-dependent vector to each token embedding. Without them, the model would be **permutation invariant**: "dog bites man" = "man bites dog."

### Q10: Explain the training objective of GPT.

**A:** GPT is trained with **Causal Language Modeling**: maximize the probability of each token given all previous tokens. The loss is cross-entropy over the next-token distribution. Because every token is a prediction target, a single sentence of length n provides n training examples, making it extremely data-efficient.

---

## 18. Further Reading & Resources

### Must-Read Papers

| Paper | Why Read It |
|---|---|
| [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) | Original Transformer |
| [GPT-1 (2018)](https://openai.com/research/language-unsupervised) | Pre-train + fine-tune paradigm |
| [GPT-2 (2019)](https://openai.com/research/better-language-models) | Scale and zero-shot emergent behavior |
| [GPT-3 (2020)](https://arxiv.org/abs/2005.14165) | In-context learning, massive scale |
| [InstructGPT (2022)](https://arxiv.org/abs/2203.02155) | RLHF alignment |
| [Chinchilla (2022)](https://arxiv.org/abs/2203.15556) | Optimal scaling laws |
| [GPT-4 Technical Report (2023)](https://arxiv.org/abs/2303.08774) | Latest capabilities |

### Interactive Visualizations

- [The Illustrated Transformer – Jay Alammar](https://jalammar.github.io/illustrated-transformer/)
- [The Illustrated GPT-2 – Jay Alammar](https://jalammar.github.io/illustrated-gpt2/)
- [BertViz – Attention Visualization](https://github.com/jessevig/bertviz)
- [Tiktokenizer (tokenization explorer)](https://tiktokenizer.vercel.app/)
- [LLM Visualization (3D walkthrough)](https://bbycroft.net/llm)

### Video Lectures

- [Andrej Karpathy — Let's build GPT from scratch (YouTube)](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [Stanford CS224N – NLP with Deep Learning](https://web.stanford.edu/class/cs224n/)
- [3Blue1Brown — Attention in Transformers](https://www.youtube.com/watch?v=eMlx5fFNoYc)

### Code to Study

- [nanoGPT by Karpathy](https://github.com/karpathy/nanoGPT) — minimal, clean GPT implementation in ~300 lines
- [Hugging Face Transformers (GPT-2)](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py)

---

## Quick Reference: GPT Forward Pass Summary

```
Input: token sequence [t_1, ..., t_n]
          │
          ▼
1. Tokenize → integer IDs
          │
          ▼
2. Token Embed: E[token_ids]     shape: (n, d)
   + Positional Embed: P[0..n]   shape: (n, d)
   ─────────────────────────────────────────
   Combined: X                    shape: (n, d)
          │
          ▼
3. For each of N Transformer blocks:
   a. LN → Masked Multi-Head Attention → residual add
   b. LN → FFN (expand 4×, GELU, project back) → residual add
          │
          ▼
4. Final Layer Norm
          │
          ▼
5. Linear (LM Head): d → V        shape: (n, V)
          │
          ▼
6. Softmax → P(next token | context)
          │
          ▼
7. Sample / Argmax → next token id → decode to text
```

---

*Study guide compiled for GPT Architecture interview preparation. All diagrams are ASCII art for portability.*
