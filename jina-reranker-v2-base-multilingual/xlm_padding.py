# This implementation was adapted from https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/modules/block.py
# Commit id: c94cd09744d20f0ac587a351ff6ff2e8ad11ae1b

# Previously adapted from https://github.com/mlcommons/training_results_v1.1/blob/main/NVIDIA/benchmarks/bert/implementations/pytorch/padding.py

import torch
import torch.nn.functional as F
from einops import rearrange, repeat


class IndexFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, indices):
        ctx.save_for_backward(indices)
        assert input.ndim >= 2
        ctx.first_axis_dim, other_shape = input.shape[0], input.shape[1:]
        second_dim = other_shape.numel()
        # TD [2022-03-04] For some reason torch.gather is a bit faster than indexing.
        # return input[indices]
        return torch.gather(
            rearrange(input, "b ... -> b (...)"), 0, repeat(indices, "z -> z d", d=second_dim)
        ).reshape(-1, *other_shape)

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        assert grad_output.ndim >= 2
        other_shape = grad_output.shape[1:]
        grad_output = rearrange(grad_output, "b ... -> b (...)")
        grad_input = torch.zeros(
            [ctx.first_axis_dim, grad_output.shape[1]],
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        # TD [2022-03-04] For some reason torch.scatter is a bit faster than indexing.
        # grad_input[indices] = grad_output
        grad_input.scatter_(0, repeat(indices, "z -> z d", d=grad_output.shape[1]), grad_output)
        return grad_input.reshape(ctx.first_axis_dim, *other_shape), None


index_first_axis = IndexFirstAxis.apply


class IndexPutFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, indices, first_axis_dim):
        ctx.save_for_backward(indices)
        assert indices.ndim == 1
        assert values.ndim >= 2
        output = torch.zeros(
            first_axis_dim, *values.shape[1:], device=values.device, dtype=values.dtype
        )
        # TD [2022-03-04] For some reason torch.scatter is a bit faster than indexing.
        output[indices] = values
        # output.scatter_(0, repeat(indices, 'z -> z d', d=values.shape[1]), values)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        # TD [2022-03-04] For some reason torch.gather is a bit faster than indexing.
        grad_values = grad_output[indices]
        # grad_values = torch.gather(grad_output, 0, repeat(indices, 'z -> z d', d=grad_output.shape[1]))
        return grad_values, None, None


index_put_first_axis = IndexPutFirstAxis.apply


class IndexFirstAxisResidual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, indices):
        ctx.save_for_backward(indices)
        assert input.ndim >= 2
        ctx.first_axis_dim, other_shape = input.shape[0], input.shape[1:]
        second_dim = other_shape.numel()
        # TD [2022-03-04] For some reason torch.gather is a bit faster than indexing.
        output = input[indices]
        # We don't want to reshape input (b ... -> b (...)) since it could change the channel_last
        # memory format to channel_first. In other words, input might not be contiguous.
        # If we don't detach, Pytorch complains about output being a view and is being modified inplace
        return output, input.detach()

    @staticmethod
    def backward(ctx, grad_output, grad_residual):
        (indices,) = ctx.saved_tensors
        assert grad_output.ndim >= 2
        other_shape = grad_output.shape[1:]
        assert grad_residual.shape[1:] == other_shape
        grad_input = grad_residual
        # grad_input[indices] += grad_output
        indices = indices.reshape(indices.shape[0], *((1,) * (grad_output.ndim - 1)))
        indices = indices.expand_as(grad_output)
        grad_input.scatter_add_(0, indices, grad_output)
        return grad_input.reshape(ctx.first_axis_dim, *other_shape), None


index_first_axis_residual = IndexFirstAxisResidual.apply

# @torch._dynamo.disable
# def mark_unpadded_dynamic(unpadded, hidden_states):
#     torch._dynamo.mark_dynamic(
#         unpadded,
#         0,
#         min=8,
#         max=hidden_states.size(0) * hidden_states.size(1)
#     )

# def unpad_input(hidden_states, attention_mask):
#     """
#     Arguments:
#         hidden_states: (batch, seqlen, ...)
#         attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.
#     Return:
#         hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
#         indices: (total_nnz), the indices of non-masked tokens from the flattened input sequence.
#         cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
#         max_seqlen_in_batch: int
#     """
#     seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
#     indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
#     max_seqlen_in_batch = seqlens_in_batch.max().item()
#     cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
#     # TD [2022-03-04] We don't want to index with a bool mask, because Pytorch will expand the
#     # bool mask, then call nonzero to get the indices, then index with those. The indices is @dim
#     # times larger than it needs to be, wasting memory. It's faster and more memory-efficient to
#     # index with integer indices. Moreover, torch's index is a bit slower than it needs to be,
#     # so we write custom forward and backward to make it a bit faster.
#     unpadded = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices)
#     mark_unpadded_dynamic(unpadded, hidden_states)
#     return (
#         unpadded,
#         indices,
#         cu_seqlens,
#         max_seqlen_in_batch,
#     )

@torch._dynamo.disable
def mark_dynamic_hidden_states(tensor, batch, seqlen):
    # Mark dimension 0 as dynamic (for the flattened token count)
    # Adjust min/max as appropriate for your expected range.
    torch._dynamo.mark_dynamic(tensor, 0, min=batch * 8, max=batch * 512)
    return tensor

def unpad_input(hidden_states, key_padding_mask):
    # Assume hidden_states.shape is (batch, seqlen, hidden)
    batch, seqlen = hidden_states.shape[:2]
    # Use fixed indices for simplicity.
    indices = torch.arange(batch * seqlen, device=hidden_states.device)
    cu_seqlens = torch.arange(0, (batch + 1) * seqlen, seqlen, device=hidden_states.device, dtype=torch.int32)
    # Flatten hidden_states to (batch * seqlen, hidden)
    unpadded = hidden_states.view(-1, hidden_states.shape[-1])
    # Mark dynamic now so that when unpadded is passed into a layer, it doesn't trigger a recompile:
    unpadded = mark_dynamic_hidden_states(unpadded, batch, seqlen)
    return unpadded, indices, cu_seqlens, seqlen

# def unpad_input(hidden_states, key_padding_mask):
#     # Assume hidden_states.shape is (batch, seqlen, hidden)
#     batch, seqlen = hidden_states.shape[:2]
    
#     # Instead of using nonzero, we can use cumsum for sequence lengths
#     seqlens_in_batch = key_padding_mask.sum(dim=-1, dtype=torch.int32)
#     max_seqlen_in_batch = seqlens_in_batch.max().item()
    
#     # Create cumulative sequence lengths
#     cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    
#     # Create indices based on the mask directly
#     indices = torch.arange(batch * seqlen, device=hidden_states.device)[key_padding_mask.flatten()]
    
#     # Get the valid tokens using masked_select instead of indexing
#     valid_tokens = torch.masked_select(
#         hidden_states.view(-1, hidden_states.shape[-1]),
#         key_padding_mask.flatten().unsqueeze(-1)
#     ).view(-1, hidden_states.shape[-1])
    
#     return valid_tokens, indices, cu_seqlens, max_seqlen_in_batch


def unpad_input_for_concatenated_sequences(hidden_states, attention_mask_in_length):
    """
    Supports concatenating short samples in one sequence. The attention_mask_in_length is utilized to mask other short samples. It helps efficient training of variant lengths-based samples (e.g., the supervised fine-tuning task in large language model).
    The motivation for this function is explained [here](https://github.com/Dao-AILab/flash-attention/issues/432#issuecomment-1668822286).

    For example, if batch = 3 and seqlen = 6, the attention_mask_in_length is:
        ```
        [
          [2, 3, 0, 0, 0, 0],
          [3, 2, 0, 0, 0, 0],
          [6, 0, 0, 0, 0, 0]
        ]
        ```
    , which refers to the 3D-attention mask:
        ```
        [
          [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 1]
          ],
          [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 1]
          ],
          [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1]
          ]
        ]
        ```.

    Arguments:
        hidden_states: (batch, seqlen, ...)
        attention_mask_in_length: (batch, seqlen), int, a nonzero number (e.g., 1, 2, 3, etc.) means length of concatenated sequence in b-th batch, and 0 means none.
    Return:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz), the indices of non-masked tokens from the flattened input sequence.
        cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
        max_seqlen_in_batch: int
    """
    length = attention_mask_in_length.sum(dim=-1)
    seqlen = attention_mask_in_length.size(-1)
    attention_mask_2d = torch.arange(seqlen, device=length.device, dtype=length.dtype).expand(len(length),
                                                                                              seqlen) < length.unsqueeze(
        1)
    real_indices_idx = torch.nonzero(attention_mask_in_length.flatten(), as_tuple=False).flatten()
    seqlens_in_batch = attention_mask_in_length.flatten()[real_indices_idx]
    indices = torch.nonzero(attention_mask_2d.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    # TD [2022-03-04] We don't want to index with a bool mask, because Pytorch will expand the
    # bool mask, then call nonzero to get the indices, then index with those. The indices is @dim
    # times larger than it needs to be, wasting memory. It's faster and more memory-efficient to
    # index with integer indices. Moreover, torch's index is a bit slower than it needs to be,
    # so we write custom forward and backward to make it a bit faster.
    return (
        index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices),
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def pad_input(hidden_states, indices, batch, seqlen):
    """
    Arguments:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens
        indices: (total_nnz), indices of non-masked tokens
        batch: int, batch size
        seqlen: int, maximum sequence length for the padded sequence.
    Return:
        hidden_states: (batch, seqlen, ...)
    """
    # Reassemble the padded tensor from flattened tokens.
    output = index_put_first_axis(hidden_states, indices, batch * seqlen)
    output = rearrange(output, "(b s) ... -> b s ...", b=batch)
    # Mark the sequence dimension (dim=1) as dynamic.
    output = _mark_dynamic_pad_output(output, seqlen)
    return output

@torch._dynamo.disable
def _mark_dynamic_pad_output(output, seqlen):
    # Instead of using the current seqlen as max, use the overall possible max.
    # For example, if BUCKETS = [16, 32, …, 512]:
    torch._dynamo.mark_dynamic(output, 1, min=8, max=512)
    return output