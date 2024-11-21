# Sequence Transformer Design Decisions

## Sequence Length
Options:
  - [x] 8 previous numbers
  - [ ] 16 previous numbers
  - [ ] Variable length with padding
  - Rationale: Starting with fixed length of 8 for simplicity

## Embedding Approach
Options:
  - [ ] Random initialized learnable embeddings
  - [ ] Positional/frequency-based initialization
  - [ ] Number-theoretic initialization (modulo, factors, etc)
  - [ ] Fixed binary encoding (8 bits)
  - [x] Fixed (integer/max_integer, position/max_position) tuple
    - Embedding Dimension is 2 (number + position)
  - Rationale: Directly encode number and position information

## Model Architecture
Number of transformer layers:
  - [x] 2-3 layers
  - [ ] 4-6 layers
  - [ ] 6+ layers
  - Rationale: Small number of layers sufficient for simple patterns

Number of attention heads:
  - [ ] 1 head
  - [x] 2 heads
  - [ ] 4 heads
  - [ ] 8+ heads
  - Rationale: Few heads needed due to small embedding dimension (2)

Feed-forward dimension:
  - [x] 32
  - [ ] 64
  - [ ] 128+
  - Rationale: Small FFN suitable for 2-dim embeddings

Dropout rates:
  - [x] 0.0 (no dropout)
  - [ ] 0.1
  - [ ] 0.2+
  - Rationale: No dropout needed since we're modeling exact mathematical patterns without noise

## Output Format
Options:
  - [ ] Direct integer prediction (regression)
  - [x] Classification with 256 classes
  - Rationale: More natural for transformers, clear probability distribution over possible next values

## Loss Function
Options:
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
  - Suggested value: α = 0.1
  - Example: if true=10, prediction distribution peaks at 100:
    - Basic cross-entropy would only care about wrong classification
    - Distance weight adds 90× more penalty compared to peaking at 11

## Loss Handling for Masked Positions
Options:
  - [ ] Ignore masked positions in loss calculation
  - [x] Include all positions in loss calculation
    - Normalize by sequence length
  - [ ] Weighted combination based on position
  - Rationale: Including all positions helps model learn from partial sequences

## Activation Functions
Options:
  - [ ] ReLU
  - [x] GELU
  - [ ] Swish
  - Rationale: GELU is standard in modern transformers, provides smoother gradients

## Layer Normalization
Options:
  - [ ] No layer normalization
  - [ ] Pre-norm (normalize before attention/FFN)
  - [x] Post-norm (normalize after attention/FFN)
    - epsilon = 1e-5
    - Learn scale and bias parameters
  - Rationale: Post-norm is the original transformer design, works well for smaller models

## Training Approach
Options:
  - [x] Train on all prefixes (causal modeling/masking)
  - [ ] Train only on full-length sequences
  - Rationale: Maximizes use of training data, helps model learn patterns of varying lengths

## Training Parameters
Batch size:
  - [x] 32
  - [ ] 64
  - [ ] 128+
  - Rationale: Small batch size sufficient for exact patterns

Learning rate:
  - [ ] Constant learning rate
  - [x] Linear warmup + cosine decay
    - Warmup steps: 5000
    - Min learning rate: 1e-6
  - [ ] Step decay
  - [ ] Exponential decay
  - Rationale: Warmup helps with initial training stability, cosine decay provides smooth learning rate reduction

Optimizer:
  - [x] AdamW
    - β1 = 0.9 (momentum)
    - β2 = 0.999 (RMSprop factor)
    - weight_decay = 0.01
    - ε = 1e-8
  - [ ] Adam
  - [ ] SGD
  - Rationale: AdamW is standard for transformers, weight decay may help with generalization

Number of epochs:
  - [x] 50-100
  - [ ] 100-500
  - [ ] 500+
  - Rationale: Since patterns are exact, model should converge relatively quickly

## Early Stopping Criteria
Options:
  - [x] Validation loss plateau
  - [ ] Validation accuracy plateau
  - [ ] Combined criteria (loss and accuracy)
  - Rationale: Monitoring validation loss is standard and directly relates to model's generalization

Implementation details:
  - Patience: 10 epochs
  - Minimum delta: 0.001 (minimum change to qualify as an improvement)
  - Save best weights (lowest validation loss) during training
  - Restore best weights after stopping

## Curriculum Learning Strategy
Phase 1: Simple recurrence relations
  - Simple recurrence relations on the past elements

  - Order 0 (e.g., [1,1,1,...])
  - Order 1 (e.g., [2,4,6,...])
  - Order 2 (e.g., [1,0,1,0,...])

Phase 2: Nested recurrence relations
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

Phase 3: Combined and transformed recurrence relations
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

Training Phase Transitions
  - Options:
    - [ ] Strict jump between phases
    - [x] Overlapping/gradual transitions
    - Rationale: Gradual transitions help the model adapt to complexity without losing learned patterns

Implementation details:
  - Gradually increase the proportion of complex patterns in the training data
  - Use a linear schedule over a fixed number of epochs to transition between phases

## Generation Strategy
Options:
  - [x] Cached/Incremental decoding
  - [ ] Naive sequential generation
  - [ ] Parallel generation
  - Rationale: Caching previous key/value states avoids redundant computation

## Generation Decoding Strategy
Options:
  - [x] Greedy decoding
  - [ ] Beam search (width=2,3,...)
  - [ ] Temperature sampling
  - Rationale: For deterministic sequences, greedy decoding should suffice since there should be one clear "right" answer

## Gradient Clipping
Options:
  - [ ] No gradient clipping
  - [x] Global norm clipping (max_norm=1.0)
  - [ ] Value clipping (clip_value=±1.0)
  - Rationale: Global norm clipping is standard practice for transformers, helps prevent exploding gradients

## Validation
No explicit train/validation/test split needed
  - Rationale: Training data is generated on-the-fly with parameters from fixed distributions
For validation/testing:
  - Generate new sequences using same parameter distributions
  - Evaluate model on these fresh sequences
  - No risk of data leakage since generation is deterministic given parameters

## Validation Frequency
Options:
  - [ ] Every epoch
  - [x] Every 1000 training steps
  - [ ] Every N minutes
  - [ ] Dynamic/adaptive frequency
  - Rationale: Fixed step interval provides consistent monitoring without too much overhead

Validation process:
- Generate 100 fresh sequences
- Evaluate full sequence prediction accuracy
- Track metrics:
  - Classification accuracy (vs random baseline: 1/256 ≈ 0.39%)
  - Mean numerical distance error (vs random baseline: 85.33 for uniform distribution over [0,255])
  - Loss value (vs random baseline: -ln(1/256) ≈ 5.55 for uniform distribution)
- Compare all metrics against random predictor baselines
- Log metrics and baseline comparisons to tensorboard

## Memory Estimation
Embedding Layer:
  - Memory = num_embeddings * embedding_dim * 4 bytes (for float32)
Transformer Layers:
  - Memory per layer = 2 * (embedding_dim * num_heads * seq_length * 4 bytes) + (feed_forward_dim * seq_length * 4 bytes)
  - Total memory = num_layers * memory per layer
Batch Size:
  - Memory = batch_size * seq_length * embedding_dim * 4 bytes
Intermediate Activations:
  - Additional memory required for backpropagation, typically 2-3x forward pass memory

## GPU Specifications
Ensure model fits within 8 GB memory limit
AMD Radeon RX 6600
  - Infinity Cache: 32 MB
  - Max Memory Size: 8 GB GDDR6
  - Memory Bandwidth: Up to 224 GB/s

