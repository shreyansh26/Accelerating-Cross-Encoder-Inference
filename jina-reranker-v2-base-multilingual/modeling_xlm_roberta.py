# This implementation was adopted from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/models/bert.py
# Commit id: abbc1311731867310635f9edc2a9ec18317c8c48
# Copyright (c) 2022, Tri Dao.
# This BERT implementation is based on our MLPerf 2.0 and MLPerf 2.1 BERT implementation.
# https://github.com/mlcommons/training_results_v2.0/blob/main/HazyResearch/benchmarks/bert/implementations/pytorch/modeling.py
# https://github.com/mlcommons/training_results_v2.1/blob/main/Azure-HazyResearch/benchmarks/bert/implementations/ND96amsr_A100_v4/modeling.py

# Inspired by https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py

import importlib.util
import logging
import re
from collections import OrderedDict
from collections.abc import Sequence
from functools import partial
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from einops import rearrange
from transformers import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import MaskedLMOutput,SequenceClassifierOutput
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaLMHead

from transformers.models.bert.modeling_bert import (
    BaseModelOutputWithPoolingAndCrossAttentions,
    BertForPreTrainingOutput,
)

from typing import List, Optional, Tuple, Union

from .xlm_padding import (
    index_first_axis,
    index_first_axis_residual,
    pad_input,
    unpad_input,
)
from .configuration_xlm_roberta import XLMRobertaFlashConfig
from .block import Block
from .embedding import XLMRobertaEmbeddings
from .mha import MHA
from .mlp import FusedMLP, Mlp

try:
    from flash_attn.ops.fused_dense import FusedDense
except ImportError:
    FusedDense = None

try:
    from flash_attn.ops.triton.layer_norm import layer_norm_fn
except ImportError:
    layer_norm_fn = None


try:
    from flash_attn.losses.cross_entropy import CrossEntropyLoss
except ImportError:
    CrossEntropyLoss = torch.nn.CrossEntropyLoss

try:
    from tqdm.autonotebook import trange
except ImportError:
    trange = None


logger = logging.getLogger(__name__)


def get_use_flash_attn(config: XLMRobertaFlashConfig):
    if not getattr(config, "use_flash_attn", False):
        return False
    if not torch.cuda.is_available():
        return False
    if importlib.util.find_spec("flash_attn") is None:
        logger.warning(
            'flash_attn is not installed. Using PyTorch native attention implementation.'
        )
        return False
    return True


def create_mixer_cls(config, cross_attn=False, return_residual=False):
    use_flash_attn = get_use_flash_attn(config)
    fused_bias_fc = getattr(config, "fused_bias_fc", False)

    mixer_cls = partial(
        MHA,
        num_heads=config.num_attention_heads,
        cross_attn=cross_attn,
        dropout=config.attention_probs_dropout_prob,
        causal=False,
        fused_bias_fc=fused_bias_fc,
        use_flash_attn=use_flash_attn,
        return_residual=return_residual,
    )
    return mixer_cls


def create_mlp_cls(config, layer_idx=None, return_residual=False):
    inner_dim = config.intermediate_size
    fused_mlp = getattr(config, "fused_mlp", False)
    if fused_mlp:
        assert config.hidden_act in ["gelu_new", "gelu_fast", "gelu_pytorch_tanh"], (
            "fused_mlp only " "supports approximate gelu"
        )
    if not fused_mlp:
        approximate = (
            "tanh"
            if config.hidden_act in ["gelu_new", "gelu_fast", "gelu_pytorch_tanh"]
            else "none"
        )
        mlp_cls = partial(
            Mlp,
            hidden_features=inner_dim,
            activation=partial(F.gelu, approximate=approximate),
            return_residual=return_residual,
        )
    else:
        if FusedMLP is None:
            raise ImportError("fused_dense is not installed")
        mlp_checkpoint_lvl = getattr(config, "mlp_checkpoint_lvl", 0)
        # mlp_checkpoint_lvl could be a list, which contains the checkpoint_lvl for each layer
        if isinstance(mlp_checkpoint_lvl, Sequence):
            assert layer_idx is not None
            mlp_checkpoint_lvl = mlp_checkpoint_lvl[layer_idx]
        mlp_cls = partial(
            FusedMLP,
            hidden_features=inner_dim,
            checkpoint_lvl=mlp_checkpoint_lvl,
            return_residual=return_residual,
        )
    return mlp_cls


def create_block(config, layer_idx=None):
    last_layer_subset = getattr(config, "last_layer_subset", False)
    cross_attn = last_layer_subset and layer_idx == config.num_hidden_layers - 1
    # TD [2022-12-19]: For cross attention (last layer), we actually want to return the
    # residual x_kv, not residual x. But it's annoying to change the API (and it only affects
    # one layer) so we just choose not to return residual in this case.
    return_residual = not cross_attn
    mixer_cls = create_mixer_cls(config, cross_attn, return_residual=return_residual)
    mlp_cls = create_mlp_cls(config, layer_idx, return_residual=return_residual)
    norm_cls = partial(nn.LayerNorm, eps=config.layer_norm_eps)
    block = Block(
        config.hidden_size,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        prenorm=False,
        resid_dropout1=config.hidden_dropout_prob,
        resid_dropout2=config.hidden_dropout_prob,
        fused_dropout_add_ln=getattr(config, "fused_dropout_add_ln", False),
        return_residual=return_residual,
    )
    return block


# https://github.com/huggingface/transformers/blob/7032e0203262ebb2ebf55da8d2e01f873973e835/src/transformers/models/bert/modeling_bert.py#L748
def _init_weights(module, initializer_range=0.02):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=initializer_range)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)
        if module.padding_idx is not None:
            nn.init.zeros_(module.weight[module.padding_idx])


class XLMRobertaEncoder(nn.Module):
    def __init__(self, config: XLMRobertaFlashConfig):
        super().__init__()
        self.use_flash_attn = get_use_flash_attn(config)
        self.layers = nn.ModuleList(
            [create_block(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self._grad_checkpointing = False

    @property
    def gradient_checkpointing(self):
        return self._grad_checkpointing

    @gradient_checkpointing.setter
    def gradient_checkpointing(self, value):
        self._grad_checkpointing = value

    def forward(self, hidden_states, key_padding_mask=None, subset_mask=None):
        """If subset_mask is not None, we only want output for the subset of the sequence.
        This means that we only compute the last layer output for these tokens.
        subset_mask: (batch, seqlen), dtype=torch.bool
        """
        if key_padding_mask is None or not self.use_flash_attn:
            mixer_kwargs = (
                {"key_padding_mask": key_padding_mask.bool()}
                if key_padding_mask is not None
                else None
            )
            for layer in self.layers:
                if self._grad_checkpointing:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        layer,
                        hidden_states,
                        use_reentrant=False,
                        mixer_kwargs=mixer_kwargs,
                    )
                else:
                    hidden_states = layer(hidden_states, mixer_kwargs=mixer_kwargs)
            if subset_mask is not None:
                hidden_states = hidden_states[subset_mask]
        else:
            batch, seqlen = hidden_states.shape[:2]
            hidden_states, indices, cu_seqlens, max_seqlen_in_batch = unpad_input(
                hidden_states, key_padding_mask
            )
            mixer_kwargs = {"cu_seqlens": cu_seqlens, "max_seqlen": max_seqlen_in_batch}
            if subset_mask is None:
                for layer in self.layers:
                    if self._grad_checkpointing:
                        hidden_states = torch.utils.checkpoint.checkpoint(
                            layer,
                            hidden_states,
                            use_reentrant=False,
                            mixer_kwargs=mixer_kwargs,
                        )
                    else:
                        hidden_states = layer(hidden_states, mixer_kwargs=mixer_kwargs)
                hidden_states = pad_input(hidden_states, indices, batch, seqlen)
            else:
                for layer in self.layers[:-1]:
                    if self._grad_checkpointing:
                        hidden_states = torch.utils.checkpoint.checkpoint(
                            layer,
                            hidden_states,
                            use_reentrant=False,
                            mixer_kwargs=mixer_kwargs,
                        )
                    else:
                        hidden_states = layer(hidden_states, mixer_kwargs=mixer_kwargs)
                if key_padding_mask is not None:
                    subset_idx = torch.nonzero(
                        subset_mask[key_padding_mask], as_tuple=False
                    ).flatten()
                    subset_seqlens = (subset_mask & key_padding_mask).sum(
                        dim=-1, dtype=torch.int32
                    )
                    subset_cu_seqlens = F.pad(
                        torch.cumsum(subset_seqlens, dim=0, dtype=torch.torch.int32),
                        (1, 0),
                    )
                else:
                    subset_idx = torch.nonzero(subset_mask, as_tuple=False).flatten()
                    subset_seqlens = subset_mask.sum(dim=-1, dtype=torch.int32)
                    subset_cu_seqlens = F.pad(
                        torch.cumsum(subset_seqlens, dim=0, dtype=torch.torch.int32),
                        (1, 0),
                    )
                hidden_states_subset, hidden_states = index_first_axis_residual(
                    hidden_states, subset_idx
                )
                # It's ok to set max_seqlen_q to be much larger
                mixer_kwargs = {
                    "x_kv": hidden_states,
                    "cu_seqlens": subset_cu_seqlens,
                    "max_seqlen": max_seqlen_in_batch,
                    "cu_seqlens_k": cu_seqlens,
                    "max_seqlen_k": max_seqlen_in_batch,
                }
                if self._grad_checkpointing:
                    torch.utils.checkpoint.checkpoint(
                        self.layers[-1],
                        hidden_states_subset,
                        use_reentrant=False,
                        mixer_kwargs=mixer_kwargs,
                    )
                else:
                    hidden_states = self.layers[-1](
                        hidden_states_subset, mixer_kwargs=mixer_kwargs
                    )
        return hidden_states


class XLMRobertaPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        fused_bias_fc = getattr(config, "fused_bias_fc", False)
        if fused_bias_fc and FusedDense is None:
            raise ImportError("fused_dense is not installed")
        linear_cls = nn.Linear if not fused_bias_fc else FusedDense
        self.dense = linear_cls(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states, pool=True):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0] if pool else hidden_states
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class XLMRobertaPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        fused_bias_fc = getattr(config, "fused_bias_fc", False)
        if fused_bias_fc and FusedDense is None:
            raise ImportError("fused_dense is not installed")
        self.fused_dropout_add_ln = getattr(config, "fused_dropout_add_ln", False)
        if self.fused_dropout_add_ln and layer_norm_fn is None:
            raise ImportError("Triton is not installed")
        linear_cls = nn.Linear if not fused_bias_fc else FusedDense
        self.dense = linear_cls(config.hidden_size, config.hidden_size)
        approximate = (
            "tanh"
            if config.hidden_act in ["gelu_new", "gelu_fast", "gelu_pytorch_tanh"]
            else "none"
        )
        self.transform_act_fn = nn.GELU(approximate=approximate)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        if not self.fused_dropout_add_ln:
            hidden_states = self.layer_norm(hidden_states)
        else:
            hidden_states = layer_norm_fn(
                hidden_states,
                self.layer_norm.weight,
                self.layer_norm.bias,
                eps=self.layer_norm.eps,
            )
        return hidden_states


class XLMRobertaLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        fused_bias_fc = getattr(config, "fused_bias_fc", False)
        if fused_bias_fc and FusedDense is None:
            raise ImportError("fused_dense is not installed")
        linear_cls = nn.Linear if not fused_bias_fc else FusedDense

        self.transform = XLMRobertaPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = linear_cls(config.hidden_size, config.vocab_size, bias=True)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class XLMRobertaPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = XLMRobertaLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class XLMRobertaPreTrainedModel(PreTrainedModel):
    """An abstract class to handle weights initialization and
    a simple interface for dowloading and loading pretrained models.
    """

    config_class = XLMRobertaFlashConfig
    base_model_prefix = "roberta"
    supports_gradient_checkpointing = True

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, XLMRobertaEncoder):
            module.gradient_checkpointing = value

    @classmethod
    def from_pretrained(
        cls,
        *args,
        **kwargs,
    ):
        if not 'torch_dtype' in kwargs:
            kwargs['torch_dtype'] = 'auto'
        return super().from_pretrained(*args, **kwargs)



class XLMRobertaModel(XLMRobertaPreTrainedModel):
    def __init__(self, config: XLMRobertaFlashConfig, add_pooling_layer=True):
        super().__init__(config)
        self.pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
        if config.vocab_size % self.pad_vocab_size_multiple != 0:
            config.vocab_size += self.pad_vocab_size_multiple - (
                config.vocab_size % self.pad_vocab_size_multiple
            )
        self.fused_dropout_add_ln = getattr(config, "fused_dropout_add_ln", False)
        if self.fused_dropout_add_ln and layer_norm_fn is None:
            raise ImportError("Triton is not installed")
        assert config.hidden_act in [
            "gelu",
            "gelu_new",
            "gelu_fast",
            "gelu_pytorch_tanh",
        ]

        self.embeddings = XLMRobertaEmbeddings(
            config.hidden_size,
            config.vocab_size,
            config.max_position_embeddings if config.position_embedding_type == 'absolute' else -1,
            config.type_vocab_size,
            padding_idx=config.pad_token_id,
        )
        self.emb_drop = nn.Dropout(config.hidden_dropout_prob)
        self.emb_ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.encoder = XLMRobertaEncoder(config)
        self.pooler = XLMRobertaPooler(config) if add_pooling_layer else None

        self.apply(partial(_init_weights, initializer_range=config.initializer_range))


    @torch.inference_mode()
    def encode(
        self: 'XLMRobertaModel',
        sentences: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: Optional[bool] = None,
        output_value: str = 'sentence_embedding',
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: Optional[torch.device] = None,
        normalize_embeddings: bool = False,
        truncate_dim: Optional[int] = None,
        **tokenizer_kwargs,
    ) -> Union[List[torch.Tensor], np.ndarray, torch.Tensor]:
        """
        Computes sentence embeddings
        Args:
            sentences(`str` or `List[str]`):
                Sentence or sentences to be encoded
            batch_size(`int`, *optional*, defaults to 32):
                Batch size for the computation
            show_progress_bar(`bool`, *optional*, defaults to None):
                Show a progress bar when encoding sentences.
                If set to None, progress bar is only shown when
                `logger.level == logging.INFO` or `logger.level == logging.DEBUG`.
            output_value(`str`, *optional*, defaults to 'sentence_embedding'):
                Default sentence_embedding, to get sentence embeddings.
                Can be set to token_embeddings to get wordpiece token embeddings.
                Set to None, to get all output values
            convert_to_numpy(`bool`, *optional*, defaults to True):
                If true, the output is a list of numpy vectors.
                Else, it is a list of pytorch tensors.
            convert_to_tensor(`bool`, *optional*, defaults to False):
                If true, you get one large tensor as return.
                Overwrites any setting from convert_to_numpy
            device(`torch.device`, *optional*, defaults to None):
                Which torch.device to use for the computation
            normalize_embeddings(`bool`, *optional*, defaults to False):
                If set to true, returned vectors will have length 1. In that case, the
                faster dot-product (util.dot_score) instead of cosine similarity can
                be used.
            truncate_dim(`int`, *optional*, defaults to None):
                The dimension to truncate sentence embeddings to. `None` does no truncation.
            tokenizer_kwargs(`Dict[str, Any]`, *optional*, defaults to {}):
                Keyword arguments for the tokenizer
        Returns:
            By default, a list of tensors is returned.
            If convert_to_tensor, a stacked tensor is returned.
            If convert_to_numpy, a numpy matrix is returned.
        """
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.name_or_path, trust_remote_code=True
        )

        is_training = self.training
        self.eval()

        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO
                or logger.getEffectiveLevel() == logging.DEBUG
            )

        if convert_to_tensor:
            convert_to_numpy = False

        if output_value != 'sentence_embedding':
            convert_to_tensor = False
            convert_to_numpy = False

        input_was_string = False
        if isinstance(sentences, str) or not hasattr(sentences, '__len__'):
            sentences = [sentences]
            input_was_string = True

        if device is not None:
            self.to(device)

        permutation = np.argsort([-len(i) for i in sentences])
        inverse_permutation = np.argsort(permutation)
        sentences = [sentences[idx] for idx in permutation]

        tokenizer_kwargs['padding'] = tokenizer_kwargs.get('padding', True)
        tokenizer_kwargs['max_length'] = tokenizer_kwargs.get(
            'max_length', self.tokenizer.init_kwargs.get('model_max_length', 8192)
        )
        tokenizer_kwargs['truncation'] = tokenizer_kwargs.get('truncation', True)

        all_embeddings = []

        if trange is not None:
            range_iter = trange(
                0,
                len(sentences),
                batch_size,
                desc="Encoding",
                disable=not show_progress_bar,
            )
        else:
            range_iter = range(0, len(sentences), batch_size)

        for i in range_iter:
            encoded_input = self.tokenizer(
                sentences[i : i + batch_size],
                return_tensors='pt',
                **tokenizer_kwargs,
            ).to(self.device)
            token_embs = self.forward(**encoded_input)[0]

            # Accumulate in fp32 to avoid overflow
            token_embs = token_embs.float()

            if output_value == 'token_embeddings':
                raise NotImplementedError
            elif output_value is None:
                raise NotImplementedError
            else:
                if self.config.emb_pooler == 'cls':
                    embeddings = self.cls_pooling(
                        token_embs, encoded_input['attention_mask']
                    )
                else:
                    embeddings = self.mean_pooling(
                        token_embs, encoded_input['attention_mask']
                    )

                if normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                if convert_to_numpy:
                    embeddings = embeddings.cpu()
            all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in inverse_permutation]

        truncate_dim = truncate_dim or self.config.truncate_dim
        if truncate_dim:
            all_embeddings = self.truncate_embeddings(all_embeddings, truncate_dim)

        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

        if input_was_string:
            all_embeddings = all_embeddings[0]

        self.train(is_training)
        return all_embeddings


    def truncate_embeddings(self, embeddings, truncate_dim):
        if not self.config.matryoshka_dimensions:
            logger.warning(
                'Matryoshka embeddings are not supported, so dimension truncation will not be performed.'
            )
            return embeddings
        elif truncate_dim in self.config.matryoshka_dimensions:
            return [tensor[:truncate_dim] for tensor in embeddings]
        else:
            raise ValueError(f'The provided `truncate_dim` value of {truncate_dim} is not supported. '
                             f'Supported dimensions are {self.config.matryoshka_dimensions}.')

    def mean_pooling(
        self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )


    def cls_pooling(
        self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ):
        return token_embeddings[:,0]


    def forward(
        self,
        input_ids,
        position_ids=None,
        token_type_ids=None,
        attention_mask=None,
        masked_tokens_mask=None,
        return_dict=None,
        **kwargs,
    ):
        """If masked_tokens_mask is not None (i.e. last_layer_subset == True in XLMForPreTraining),
        we only want the output for the masked tokens. This means that we only compute the last
        layer output for these tokens.
        masked_tokens_mask: (batch, seqlen), dtype=torch.bool
        """

        if kwargs:
            for key, value in kwargs.items():
                if value is not None:
                    logger.warning(
                        'Flash attention implementation does not support kwargs: %s',
                        key,
                    )

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        hidden_states = self.embeddings(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids
        )
        # TD [2022-12:18]: Don't need to force residual in fp32
        # BERT puts embedding LayerNorm before embedding dropout.
        if not self.fused_dropout_add_ln:
            hidden_states = self.emb_ln(hidden_states)
        else:
            hidden_states = layer_norm_fn(
                hidden_states, self.emb_ln.weight, self.emb_ln.bias, eps=self.emb_ln.eps
            )
        hidden_states = self.emb_drop(hidden_states)

        if masked_tokens_mask is not None:
            batch_size, seqlen = input_ids.shape[:2]
            # We also need the first column for the CLS token
            first_col_mask = torch.zeros(
                batch_size, seqlen, dtype=torch.bool, device=input_ids.device
            )
            first_col_mask[:, 0] = True
            subset_mask = masked_tokens_mask | first_col_mask
        else:
            subset_mask = None

        sequence_output = self.encoder(
            hidden_states, key_padding_mask=attention_mask, subset_mask=subset_mask
        )

        if masked_tokens_mask is None:
            pooled_output = (
                self.pooler(sequence_output) if self.pooler is not None else None
            )
        else:
            # TD [2022-03-01]: the indexing here is very tricky.
            if attention_mask is not None:
                subset_idx = subset_mask[attention_mask]
                pool_input = sequence_output[first_col_mask[attention_mask][subset_idx]]
                sequence_output = sequence_output[
                    masked_tokens_mask[attention_mask][subset_idx]
                ]
            else:
                pool_input = sequence_output[first_col_mask[subset_mask]]
                sequence_output = sequence_output[masked_tokens_mask[subset_mask]]
            pooled_output = (
                self.pooler(pool_input, pool=False) if self.pooler is not None else None
            )

        if not return_dict:
            return sequence_output, pooled_output

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
        )


class XLMRobertaForMaskedLM(XLMRobertaPreTrainedModel):
    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `XLMRobertaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)
        self.lm_head = XLMRobertaLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.roberta.embeddings.word_embeddings

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(prediction_scores.device)
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)
            )

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (
                ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            )

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# Copied from transformers.models.roberta.modeling_roberta.RobertaClassificationHead with Roberta->XLMRoberta
class XLMRobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        fused_bias_fc = getattr(config, "fused_bias_fc", False)
        if fused_bias_fc and FusedDense is None:
            raise ImportError("fused_dense is not installed")
        linear_cls = nn.Linear if not fused_bias_fc else FusedDense
        self.dense = linear_cls(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = linear_cls(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


# Copied from transformers.models.roberta.modeling_roberta.RobertaForSequenceClassification with Roberta->XLMRoberta, ROBERTA->XLM_ROBERTA
class XLMRobertaForSequenceClassification(XLMRobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)
        self.classifier = XLMRobertaClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


    @torch.inference_mode()
    def compute_score(
        self,
        sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]],
        batch_size: int = 32,
        max_length: Optional[int] = None,
    ) -> List[float]:

        if not hasattr(self, "_tokenizer"):
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.name_or_path, trust_remote_code=True
            )

        assert isinstance(sentence_pairs, list)
        if isinstance(sentence_pairs[0], str):
            sentence_pairs = [sentence_pairs]

        all_scores = []
        for start_index in range(
            0, len(sentence_pairs), batch_size
        ):
            sentences_batch = sentence_pairs[
                start_index : start_index + batch_size
            ]
            inputs = self._tokenizer(
                sentences_batch,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=max_length,
            ).to(self.device)
            scores = (
                self.forward(**inputs, return_dict=True)
                .logits.view(
                    -1,
                )
                .float()
            )
            scores = torch.sigmoid(scores)
            all_scores.extend(scores.cpu().numpy().tolist())

        if len(all_scores) == 1:
            return all_scores[0]
        return all_scores

    def predict(
        self,
        sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]],
        batch_size: int = 32,
        max_length: Optional[int] = None,
    ) -> List[float]:
        # used for beir evaluation
        return self.compute_score(sentence_pairs, batch_size=batch_size, max_length=max_length)

    def rerank(
        self,
        query: str,
        documents: List[str],
        batch_size: int = 32,
        max_length: int = 1024,
        max_query_length: int = 512,
        overlap_tokens: int = 80,
        top_n: Optional[int] = None,
        **kwargs,
    ):
        assert max_length >= max_query_length * 2, (
            f'max_length ({max_length}) must be greater than or equal to '
            f'max_query_length ({max_query_length}) * 2'
        )

        if not hasattr(self, "_tokenizer"):
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.name_or_path, trust_remote_code=True
            )

        # preproc of tokenization
        sentence_pairs, sentence_pairs_pids = reranker_tokenize_preproc(
            query,
            documents,
            tokenizer=self._tokenizer,
            max_length=max_length,
            max_query_length=max_query_length,
            overlap_tokens=overlap_tokens,
        )

        tot_scores = []
        with torch.no_grad():
            for k in range(0, len(sentence_pairs), batch_size):
                batch = self._tokenizer.pad(
                    sentence_pairs[k : k + batch_size],
                    padding=True,
                    max_length=max_length,
                    pad_to_multiple_of=None,
                    return_tensors="pt",
                )
                batch_on_device = {k: v.to(self.device) for k, v in batch.items()}
                scores = (
                    self.forward(**batch_on_device, return_dict=True)
                    .logits.view(
                        -1,
                    )
                    .float()
                )
                scores = torch.sigmoid(scores)
                tot_scores.extend(scores.cpu().numpy().tolist())

        # ranking
        merge_scores = [0 for _ in range(len(documents))]
        for pid, score in zip(sentence_pairs_pids, tot_scores):
            merge_scores[pid] = max(merge_scores[pid], score)

        merge_scores_argsort = np.argsort(merge_scores)[::-1]
        sorted_documents = []
        sorted_scores = []
        for mid in merge_scores_argsort:
            sorted_scores.append(merge_scores[mid])
            sorted_documents.append(documents[mid])

        top_n = min(top_n or len(sorted_documents), len(sorted_documents))

        return [
            {
                'document': sorted_documents[i],
                'relevance_score': sorted_scores[i],
                'index': merge_scores_argsort[i],
            }
            for i in range(top_n)
        ]


def reranker_tokenize_preproc(
    query: str,
    passages: List[str],
    tokenizer=None,
    max_length: int = 1024,
    max_query_length: int = 512,
    overlap_tokens: int = 80,
):
    from copy import deepcopy

    assert tokenizer is not None, "Please provide a valid tokenizer for tokenization!"
    sep_id = tokenizer.sep_token_id

    def _merge_inputs(chunk1_raw, chunk2):
        chunk1 = deepcopy(chunk1_raw)
        chunk1['input_ids'].append(sep_id)
        chunk1['input_ids'].extend(chunk2['input_ids'])
        chunk1['input_ids'].append(sep_id)
        chunk1['attention_mask'].append(chunk2['attention_mask'][0])
        chunk1['attention_mask'].extend(chunk2['attention_mask'])
        chunk1['attention_mask'].append(chunk2['attention_mask'][-1])
        if 'token_type_ids' in chunk1:
            token_type_ids = [1 for _ in range(len(chunk2['token_type_ids']) + 2)]
            chunk1['token_type_ids'].extend(token_type_ids)
        return chunk1

    # Note: the long query will be truncated to 256 tokens by default
    query_inputs = tokenizer.encode_plus(
        query, truncation=True, padding=False, max_length=max_query_length
    )

    max_passage_inputs_length = max_length - len(query_inputs['input_ids']) - 2
    # assert (
    #     max_passage_inputs_length > 100
    # ), "Your query is too long! Please make sure your query less than 500 tokens!"

    overlap_tokens_implt = min(overlap_tokens, max_passage_inputs_length // 4)

    res_merge_inputs = []
    res_merge_inputs_pids = []
    for pid, passage in enumerate(passages):
        passage_inputs = tokenizer.encode_plus(
            passage,
            truncation=False,
            padding=False,
            add_special_tokens=False,
            max_length=0,
        )
        passage_inputs_length = len(passage_inputs['input_ids'])

        if passage_inputs_length <= max_passage_inputs_length:
            qp_merge_inputs = _merge_inputs(query_inputs, passage_inputs)
            res_merge_inputs.append(qp_merge_inputs)
            res_merge_inputs_pids.append(pid)
        else:
            start_id = 0
            while start_id < passage_inputs_length:
                end_id = start_id + max_passage_inputs_length
                # make sure the length of the last chunk is `max_passage_inputs_length`
                if end_id >= passage_inputs_length:
                    sub_passage_inputs = {
                        k: v[-max_passage_inputs_length:]
                        for k, v in passage_inputs.items()
                    }
                else:
                    sub_passage_inputs = {
                        k: v[start_id:end_id] for k, v in passage_inputs.items()
                    }
                start_id = (
                    end_id - overlap_tokens_implt
                    if end_id < passage_inputs_length
                    else end_id
                )

                qp_merge_inputs = _merge_inputs(query_inputs, sub_passage_inputs)
                res_merge_inputs.append(qp_merge_inputs)
                res_merge_inputs_pids.append(pid)

    return res_merge_inputs, res_merge_inputs_pids