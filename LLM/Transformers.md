# How Transformers work ?
https://www.youtube.com/watch?v=wjZofJX0v4M

# Understanding Transformers: A Structured Overview

## 1. Introduction to GPT and Transformers

### 1.1 What GPT Stands For

-   **Generative**: Models that create new text.
-   **Pretrained**: Learned from massive datasets before fine-tuning.
-   **Transformer**: A neural network architecture powering modern AI.

### 1.2 Applications of Transformer Models

-   Text → Audio (speech synthesis)
-   Audio → Text (transcription)
-   Text → Image (e.g., DALL·E, Midjourney)
-   Language translation
-   Predicting next tokens (foundation of ChatGPT)

## 2. How Transformers Generate Text

### 2.1 Next-Token Prediction

-   Model predicts probability distribution over possible next tokens.
-   Generation loop:
    1.  Provide initial text.
    2.  Predict next token.
    3.  Sample a token.
    4.  Append & repeat.

### 2.2 Why Scaling Matters

-   Smaller models generate incoherent text.
-   Larger models produce coherent, meaningful responses.

## 3. High-Level Architecture of a Transformer

### 3.1 Tokenization

-   Text → tokens (words, subwords, punctuation)

### 3.2 Embedding Tokens as Vectors

-   Tokens represented as high-dimensional vectors.
-   Similar words → nearby vectors.

### 3.3 Attention Block

-   Determines contextual relevance.
-   Updates token representations.

### 3.4 Feed-Forward / MLP Block

-   Independently transforms each vector.
-   Learns abstract features.

### 3.5 Repeated Layers

-   Alternating attention + MLP deepens understanding.

### 3.6 Final Prediction

-   Final vector → logits → softmax → next-token probabilities.

## 4. Background Concepts

### 4.1 Machine Learning Overview

-   Models learn from data via parameters (weights).

### 4.2 Deep Learning Structure

-   Inputs & layers represented as numeric tensors.

### 4.3 Backpropagation

-   Core method for training deep networks.

## 5. Word Embeddings

### 5.1 Embedding Matrix (WE)

-   GPT‑3: 50,257 tokens × 12,288 dimensions (\~617M parameters)

### 5.2 Semantic Structure

Classic examples: - king -- man + woman ≈ queen - sushi + (Germany --
Japan) ≈ bratwurst

### 5.3 Dot Product Similarity

-   Positive: aligned
-   Zero: unrelated
-   Negative: opposite

## 6. Context Window

-   GPT‑3 context size: 2048 tokens.
-   Limits memory of earlier text.

## 7. Unembedding & Softmax

### 7.1 Unembedding Matrix

-   Similar in size to embedding matrix (\~617M params)

### 7.2 Softmax

-   Converts logits → probabilities.

### 7.3 Temperature

-   Controls randomness:
    -   Low T = predictable
    -   High T = creative but risky

## 8. Training Details

-   Each token vector predicts the next token during training.

## 9. Summary

-   Foundations: embeddings, dot products, softmax, attention, context.
-   Next key concept: **attention mechanism**.
