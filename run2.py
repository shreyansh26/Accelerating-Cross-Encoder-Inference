import torch
from ce_orig import CrossEncoder
from bench import benchmark

# Enable dynamic shape handling
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._inductor.config.triton.cudagraph_skip_dynamic_graphs=True

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
        
        for name in tokenized:
            tokenized[name] = tokenized[name].to(self.model.device)
            # Mark sequence length dimension as dynamic
            torch._dynamo.mark_dynamic(tokenized[name], 1, min=8, max=512)
        return tokenized

model = DynamicCrossEncoder(
    "jina-reranker-v2-base-multilingual",
    trust_remote_code=True,
    local_files_only=True,
    device="cuda"
)

print(model.model.dtype)
# Compile the forward pass only, not the entire model
model.model.forward = torch.compile(
    model.model.forward, 
    backend="inductor",
    mode="max-autotune",
    dynamic=True
)

benchmark(model, cuda_graph=False, print_scores=True, on_sorted_inputs=False)