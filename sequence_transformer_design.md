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
  - [ ] Fixed binary encoding (8 bits)
  - [x] (integer/max_integer, position/max_position) tuple
  - Rationale: Directly encode number and position information

## Embedding Dimension
- Options:
  - [x] 2 (number + position)
  - [ ] 8 (binary encoding)
  - [ ] 128-256 (typical transformer)
  - [ ] 32-64 (compromise)
  - Rationale: Small dimension sufficient may be sufficient for simple patterns

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
  - [ ] 1 head
  - [x] 2 heads
  - [ ] 4 heads
  - [ ] 8+ heads
  - Rationale: Few heads needed due to small embedding dimension (2)

- Feed-forward dimension:
  - [x] 32
  - [ ] 64
  - [ ] 128+
  - Rationale: Small FFN suitable for 2-dim embeddings

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

- Curriculum learning strategy:
  - Phase 1: Simple recurrence relations
    - Simple recurrence relations on the past elements

    - Order 0 (e.g., [1,1,1,...])
    - Order 1 (e.g., [2,4,6,...])
    - Order 2 (e.g., [1,0,1,0,...])

  - Phase 2: Nested recurrence relations
    - Recurrence relations where the constants themselves follow a recurrence relation

    - Arithmetic sequences with varying steps
      - Alternating steps [1,3,4,6,7,9,...] (steps: +2,+1)
      - Growing steps [1,2,4,7,11,...] (steps: +1,+2,+3,...)
      - Shrinking steps [1,5,8,10,11,...] (steps: +4,+3,+2,+1)

    - Geometric sequences
      - Simple multiplication [2,4,8,16,...]
      - Alternating factors [1,2,6,12,36,...] (factors: ×2,×3)
      - Rational factors [8,4,6,3,...] (×1/2,×3/2)

    - Multi-period patterns
      - Two-step patterns [1,1,2,2,3,3,...]
      - Three-step patterns [1,1,1,2,2,2,...]
      - Mixed periods [1,2,2,3,4,4,5,6,6,...]

  - Phase 3: Combined and transformed recurrence relations
    - Compositions
      - Apply one recurrence relation to the output of another
      - Example: fibonacci numbers mod 3
      - Example: double every third number in an arithmetic sequence

    - Transformations
      - Apply functions to terms of a recurrence relation
      - Example: floor/ceiling of geometric sequences
      - Example: alternating signs of arithmetic sequence

    - Conditional relations
      - Different rules based on previous terms
      - Example: increment if previous is even, double if odd
      - Example: switch between two different rules based on local pattern

  Rationale: Build up from simple to complex patterns helps establish basic numerical relationships first

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

## Loss Handling for Masked Positions
- Options:
  - [ ] Ignore masked positions in loss calculation
  - [x] Include all positions in loss calculation
  - [ ] Weighted combination based on position
  - Rationale: Including all positions helps model learn from partial sequences

## Validation and Testing Approach
- No explicit train/validation/test split needed
- Rationale: Training data is generated on-the-fly with parameters from fixed distributions
- For validation/testing:
  - Generate new sequences using same parameter distributions
  - Evaluate model on these fresh sequences
  - No risk of data leakage since generation is deterministic given parameters

## Generation Decoding Strategy
- Options:
  - [x] Greedy decoding
  - [ ] Beam search (width=2,3,...)
  - [ ] Temperature sampling
  - Rationale: For deterministic sequences, greedy decoding should suffice since there should be one clear "right" answer

## Activation Functions
- Options:
  - [ ] ReLU
  - [x] GELU
  - [ ] Swish
  - Rationale: GELU is standard in modern transformers, provides smoother gradients

## TBD
- Layer normalization
- Gradient clipping
- Learning rate scheduling
- Validation frequency
- Early stopping criteria
- Training phases: strict jump, overlapping/gradual transitions
- α scaling factor value/range
- Normalize the distance penalty by sequence length?
- Should distance weighting change during training phases?
- Evaluation metrics
- Model selection criteria
- Error analysis techniques
- Deployment considerations
- Hardware/software requirements
- Scalability and performance considerations

