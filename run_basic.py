from sentence_transformers import CrossEncoder
from bench import benchmark

model = CrossEncoder(
    "jinaai/jina-reranker-v2-base-multilingual",
    trust_remote_code=True,
    device="cuda",
    config_args={"use_flash_attn": True},
    max_length=512
)

benchmark(model, print_scores=False, on_sorted_inputs=False)

# With Flash Attention - Mean time: 0.2952 ± 0.0152 seconds
# Without Flash Attention - Mean time: 0.3508 ± 0.0089 seconds