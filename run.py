import torch
from ce import CrossEncoder
from bench import benchmark, test_model

torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True

model = CrossEncoder(
    "jina-reranker-v2-base-multilingual",
    trust_remote_code=True,
    local_files_only=True,
    device="cuda",
    max_length=64
)

model.model = torch.compile(model.model, backend="inductor", mode="max-autotune")

# test_model(model)
benchmark(model, print_scores=True, on_sorted_inputs=False)