# Optimizing Jina Cross Encoder with Torch Compile

## Overview
This project demonstrates optimizing the inference of the Jina Cross Encoder, namely "jinaai/jina-reranker-v2-base-multilingual", by leveraging torch.compile. The scripts compare baseline performance against a torch.compile-optimized version using a custom padding approach.

## Setup
- Python 3.8+
- PyTorch with CUDA support
- Sentence Transformers library
- Model: jinaai/jina-reranker-v2-base-multilingual

## Scripts

### run_basic.py
Runs a baseline benchmark with the standard CrossEncoder and Flash Attention enabled.

### run_torch_compile.py
Focuses on the torch.compile approach with some custom padding and torch.compile optimizations.

### run_combined.py
Compares the baseline with torch.compile optimized version.

## Implementation Details

- **Batching with custom padding**: The custom `DynamicCrossEncoder` pads tokenized inputs to a bucket length (multiples of 16), to lower the number of dynamic lengths that torch.compile has to capture.

- **torch.compile**: The model's forward function is compiled using `torch.compile` with the `inductor` backend, enabling dynamic shape handling and reducing latency.

## Speedup Analysis

The torch.compile optimized version shows significant speedups compared to the baseline (batch size 64):

| Setup                                      | Sorted Inputs (s)       | Unsorted Inputs (s)      |
| ------------------------------------------ | ----------------------- | ------------------------ |
| Base (with Flash Attention)                | 0.2721 ± 0.0060         | 0.3007 ± 0.0141          |
| torch.compile                              | 0.2104 ± 0.0035         | 0.2570 ± 0.0118          |

This reflects roughly a 20-25% reduction in inference latency under sorted inputs, with similar gains observed for unsorted inputs.

## How to Run
1. Ensure your environment has CUDA and the required libraries installed:
   - `pip install sentence-transformers torch`
2. Execute the benchmark scripts:
   - `CUDA_VISIBLE_DEVICES=0 python run_basic.py`
   - `CUDA_VISIBLE_DEVICES=0 python run_combined.py`
   - `CUDA_VISIBLE_DEVICES=0 python run_torch_compile.py`