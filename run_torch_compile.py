import torch
from sentence_transformers import CrossEncoder
from bench import benchmark

# Enable dynamic shape handling
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
# torch._inductor.config.triton.cudagraph_skip_dynamic_graphs=True


BUCKETS = list(range(16, 512, 16))
class DynamicCrossEncoder(CrossEncoder):
    def smart_batching_collate_text_only(self, batch):
        texts = [[text.strip() for text in field] for field in zip(*batch)]
        tokenized = self.tokenizer(
            *texts,
            padding=True,
            truncation="longest_first",
            return_tensors="pt",
            max_length=self.max_length
        )
        tokenized = {k: v.to(self.model.device) for k, v in tokenized.items()}

        # Pad each field to the closest bucket length (multiples of 16)
        cur_length = tokenized["input_ids"].size(1)
        bucket_length = next((b for b in BUCKETS if b >= cur_length), cur_length)
        if bucket_length > cur_length:
            diff = bucket_length - cur_length
            for key, val in tokenized.items():
                pad_value = self.tokenizer.pad_token_id if key == "input_ids" else 0
                tokenized[key] = torch.nn.functional.pad(val, (0, diff), value=pad_value)
        return tokenized

model_compile = DynamicCrossEncoder(
    "jinaai/jina-reranker-v2-base-multilingual",
    trust_remote_code=True,
    device="cuda",
    config_args={"use_flash_attn": False},
    max_length=512
)

model_compile.model.forward = torch.compile(
    model_compile.model.forward, 
    backend="inductor",
    mode="max-autotune",
    dynamic=True
)

benchmark(model_compile, print_scores=True, on_sorted_inputs=False, seed=1000)

# Sorted Inputs - Mean time: 0.2206 ± 0.0107 seconds
# Unsorted Inputs - Mean time: 0.2673 ± 0.0046 seconds