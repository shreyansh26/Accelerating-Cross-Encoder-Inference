from transformers import PretrainedConfig
import torch

class XLMRobertaFlashConfig(PretrainedConfig):
    def __init__(
            self,
            vocab_size=30522,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=2,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=1,
            bos_token_id=0,
            eos_token_id=2,
            position_embedding_type="absolute",
            use_cache=True,
            classifier_dropout=None,
            lora_adaptations=None,
            lora_rank=4,
            lora_dropout_p=0.0,
            lora_alpha=1,
            lora_main_params_trainable=False,
            load_trained_adapters=False,
            use_flash_attn=True,
            torch_dtype=None,
            emb_pooler=None,
            matryoshka_dimensions=None,
            truncate_dim=None,
            **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)


        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        self.load_trained_adapters = load_trained_adapters
        self.lora_adaptations = lora_adaptations
        self.lora_rank = lora_rank
        self.lora_dropout_p = lora_dropout_p
        self.lora_alpha = lora_alpha
        self.lora_main_params_trainable = lora_main_params_trainable
        self.use_flash_attn = use_flash_attn
        self.emb_pooler = emb_pooler
        self.matryoshka_dimensions = matryoshka_dimensions
        self.truncate_dim = truncate_dim
        if torch_dtype and hasattr(torch, torch_dtype) and type(getattr(torch, torch_dtype)) is torch.dtype:
            self.torch_dtype = getattr(torch, torch_dtype)
        else:
            self.torch_dtype = torch_dtype
