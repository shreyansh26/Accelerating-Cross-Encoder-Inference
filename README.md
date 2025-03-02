# Optimizing Cross Encoder inference with torch.compile

## Overview
This project demonstrates optimizing the inference of a Cross Encoder model, namely [jinaai/jina-reranker-v2-base-multilingual](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual), by leveraging torch.compile. The scripts compare baseline performance against a torch.compile-optimized version using a custom padding approach.

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

- **Sorted Inputs**: Sorting the inputs before batching allows the sequences in the batch to be of similar lengths hence less padding tokens to be processed.

- **torch.compile**: The model's forward function is compiled using `torch.compile` with the `inductor` backend, enabling dynamic shape handling and reducing latency.

## Speedup Analysis

The torch.compile optimized version shows significant speedups compared to the baseline (batch size 64, H100 GPU):

| Setup                                      | Sorted Inputs (s)       | Unsorted Inputs (s)      |
| ------------------------------------------ | ----------------------- | ------------------------ |
| Base (with Flash Attention)                | 0.2658 ± 0.0119         | 0.2961 ± 0.0089          |
| torch.compile                              | 0.2089 ± 0.0196         | 0.2595 ± 0.0077          |

This reflects roughly a 20-25% reduction in inference latency under sorted inputs, with similar gains observed for unsorted inputs.

## How to Run
1. Ensure your environment has CUDA and the required libraries installed:
   - `pip install sentence-transformers torch`
2. Execute the benchmark scripts:
   - `CUDA_VISIBLE_DEVICES=0 python run_basic.py`
   - `CUDA_VISIBLE_DEVICES=0 python run_combined.py`
   - `CUDA_VISIBLE_DEVICES=0 python run_torch_compile.py`
