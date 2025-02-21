import torch
from ce_orig import CrossEncoder
from bench_dataloader import benchmark

# Enable dynamic shape handling
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True

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

model = CrossEncoder(
    "jina-reranker-v2-base-multilingual",
    trust_remote_code=True,
    local_files_only=True,
    device="cuda"
)

model_compile = DynamicCrossEncoder(
    "jina-reranker-v2-base-multilingual",
    trust_remote_code=True,
    local_files_only=True,
    device="cuda"
)

model_compile.model = torch.compile(
    model_compile.model, 
    backend="inductor", 
    mode="max-autotune",
)

benchmark(model)
benchmark(model_compile)