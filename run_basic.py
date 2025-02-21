from ce_orig import CrossEncoder
from bench import benchmark

model = CrossEncoder(
    "jina-reranker-v2-base-multilingual",
    trust_remote_code=True,
    local_files_only=True,
    device="cuda"
)

benchmark(model, print_scores=True, on_sorted_inputs=True)