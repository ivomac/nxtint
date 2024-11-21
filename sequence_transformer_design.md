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
- TBD:
  - Batch size
  - Learning rate
  - Optimizer
  - Number of epochs
  - Curriculum learning strategy

## Model Architecture
- TBD:
  - Number of transformer layers
  - Number of attention heads
  - Feed-forward dimension
  - Dropout rates
