# Sequence Transformer Design Decisions

## Input/Output Format
- Options:
  - [ ] Direct integer prediction (regression)
  - [x] Classification with 256 classes
  - Rationale: More natural for transformers, clear probability distribution over possible next values

## Sequence Length
- Options:
  - [x] 8 previous numbers
  - [ ] 16 previous numbers
  - [ ] Variable length with padding
  - Rationale: Starting with fixed length of 8 for simplicity

## Embedding Approach
- Options:
  - [ ] Random initialized learnable embeddings
  - [ ] Positional/frequency-based initialization
  - [ ] Number-theoretic initialization (modulo, factors, etc)
  - [x] Fixed binary encoding (8 bits)
  - Rationale: Direct binary representation matches the natural structure of numbers

## Embedding Dimension
- Options:
  - [x] 8 (binary encoding)
  - [ ] 128-256 (typical transformer)
  - [ ] 32-64 (compromise)
  - Rationale: Using minimal 8-bit binary encoding as starting point

## Training Approach
- Options:
  - [x] Train on all prefixes (causal modeling)
  - [ ] Train only on full-length sequences
  - Rationale: Maximizes use of training data, helps model learn patterns of varying lengths

## Sequence Masking
- Options:
  - [ ] Padding with mask tokens
  - [x] Causal masking
  - Rationale: Causal masking naturally handles variable length inputs and matches the autoregressive nature of sequence prediction

## Generation Strategy
- Options:
  - [x] Cached/Incremental decoding
  - [ ] Naive sequential generation
  - [ ] Parallel generation
  - Rationale: Caching previous key/value states avoids redundant computation

## Model Architecture
- Number of transformer layers:
  - [x] 2-3 layers
  - [ ] 4-6 layers
  - [ ] 6+ layers
  - Rationale: Small number of layers sufficient for simple patterns

- Number of attention heads:
  - [x] 2 heads
  - [ ] 4 heads
  - [ ] 8+ heads
  - Rationale: Few heads needed due to small embedding dimension (8)

- Feed-forward dimension:
  - [x] 32
  - [ ] 64
  - [ ] 128+
  - Rationale: Small FFN suitable for 8-dim embeddings

- Dropout rates:
  - [x] 0.0 (no dropout)
  - [ ] 0.1
  - [ ] 0.2+
  - Rationale: No dropout needed since we're modeling exact mathematical patterns without noise

## Training Parameters
- Batch size:
  - [x] 32
  - [ ] 64
  - [ ] 128+
  - Rationale: Small batch size sufficient for exact patterns

- Learning rate:
  - [x] 1e-4
  - [ ] 1e-3
  - [ ] 5e-4
  - Rationale: Conservative learning rate for stable training

- Optimizer:
  - [x] AdamW
    - β1 = 0.9 (momentum)
    - β2 = 0.999 (RMSprop factor)
    - weight_decay = 0.01
    - ε = 1e-8
  - [ ] Adam
  - [ ] SGD
  - Rationale: AdamW is standard for transformers, weight decay may help with generalization

- Number of epochs:
  - [x] 50-100
  - [ ] 100-500
  - [ ] 500+
  - Rationale: Since patterns are exact, model should converge relatively quickly

- Curriculum learning strategy (TBD)

## Loss Function
- Options:
  - [ ] Standard cross-entropy
  - [x] Distance-weighted cross-entropy
  - [ ] Pure distance-based loss
  - Rationale: Combine classification confidence with numerical distance penalty

Distance weighting options:
  - [x] Linear: weight = |predicted - true|
  - [ ] Quadratic: weight = (predicted - true)²
  - [ ] Square root: weight = sqrt(|predicted - true|)
  - Rationale: Linear scaling provides intuitive penalty without over-emphasizing large errors

Implementation approach:
- Start with standard cross-entropy loss
- Multiply by (1 + α * distance_weight) where α is a scaling factor
- Example: if true=10, prediction distribution peaks at 100:
  - Basic cross-entropy would only care about wrong classification
  - Distance weight adds 90× more penalty compared to peaking at 11
