import torch
from ce_orig import CrossEncoder
from bench import benchmark

# Enable dynamic shape handling
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
# torch._inductor.config.triton.cudagraph_skip_dynamic_graphs=True


BUCKETS = list(range(16, 512, 16))
class DynamicCrossEncoder(CrossEncoder):
    def smart_batching_collate_text_only(self, batch):
        texts = [[] for _ in range(len(batch[0]))]
        
        for example in batch:
            for idx, text in enumerate(example):
                texts[idx].append(text.strip())

        tokenized = self.tokenizer(
            *texts,
            padding=True,  # Use True instead of max_length
            truncation='longest_first',
            return_tensors="pt",
            max_length=self.max_length
        )

        # Move each tensor to device and mark the sequence dim as dynamic
        for name in tokenized:
            tokenized[name] = tokenized[name].to(self.model.device)
            # torch._dynamo.mark_dynamic(tokenized[name], -2, min=8, max=512)

        # Pad each field to the closest bucket in BUCKETS (multiples of 16)
        # We assume that all tokenized outputs share the same sequence length.
        cur_length = tokenized["input_ids"].size(1)
        # Find the next bucket value that is >= current length; default to cur_length if none found
        bucket_length = next((b for b in BUCKETS if b >= cur_length), cur_length)
        if bucket_length > cur_length:
            for key in tokenized:
                pad_value = self.tokenizer.pad_token_id if key == "input_ids" else 0
                # Pad along the sequence length dimension (last dim)
                tokenized[key] = torch.nn.functional.pad(
                    tokenized[key],
                    (0, bucket_length - cur_length),
                    value=pad_value
                )
                torch._dynamo.mark_dynamic(tokenized[key], -2, min=8, max=512)
                # print(key, tokenized[key].shape)
        return tokenized

model_compile = DynamicCrossEncoder(
    "jina-reranker-v2-base-multilingual",
    trust_remote_code=True,
    local_files_only=True,
    device="cuda",
    config_args={"use_flash_attn": False}
)

model_compile.model.forward = torch.compile(
    model_compile.model.forward, 
    backend="inductor",
    mode="max-autotune",
    dynamic=True
)

benchmark(model_compile, print_scores=False, on_sorted_inputs=False, seed=1000)