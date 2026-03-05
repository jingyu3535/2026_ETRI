from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node

_AXIS_DEFAULT = -1
_EPS_DEFAULT = 1e-5


def _canonical_axis(axis: int, ndim: int) -> int:
    return axis if axis >= 0 else ndim + axis


def _rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    axis: int = -1,
) -> torch.Tensor:
    axis = _canonical_axis(axis, x.ndim)
    reduce_dims = tuple(range(axis, x.ndim))
    variance = x.pow(2).mean(dim=reduce_dims, keepdim=True)
    normalized = x * torch.rsqrt(variance + epsilon)

    weight_shape = [1] * axis + list(weight.shape)
    return normalized * weight.view(weight_shape)


class SimplifiedLayerNormOp(nn.Module):
    def __init__(self, axis: int = _AXIS_DEFAULT, epsilon: float = _EPS_DEFAULT):
        super().__init__()
        self.axis = axis
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return _rms_norm(x, weight, epsilon=self.epsilon, axis=self.axis)


class SkipSimplifiedLayerNormOp(nn.Module):
    def __init__(self, epsilon: float = _EPS_DEFAULT, output_count: int = 4):
        super().__init__()
        self.epsilon = epsilon
        self.output_count = output_count

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        weight: torch.Tensor,
    ):
        residual = x + skip
        normalized = _rms_norm(residual, weight, epsilon=self.epsilon, axis=-1)

        if self.output_count == 1:
            return normalized

        outputs = [residual.new_zeros(1) for _ in range(self.output_count)]
        outputs[0] = normalized
        outputs[-1] = residual
        return tuple(outputs)


class RotaryEmbeddingOp(nn.Module):
    def __init__(
        self,
        interleaved: int = 0,
        num_heads: int = 0,
        rotary_embedding_dim: int = 0,
    ):
        super().__init__()
        self.interleaved = bool(interleaved)
        self.num_heads = num_heads
        self.rotary_embedding_dim = rotary_embedding_dim

    def _infer_num_heads(self, hidden_size: int, cos_cache: torch.Tensor) -> int:
        if self.num_heads > 0:
            return self.num_heads

        inferred_head_dim = max(1, cos_cache.shape[-1] * 2)
        if hidden_size % inferred_head_dim != 0:
            raise ValueError(
                f"Cannot infer num_heads from hidden_size={hidden_size}, "
                f"inferred_head_dim={inferred_head_dim}",
            )
        return hidden_size // inferred_head_dim

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        cos_cache: torch.Tensor,
        sin_cache: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape
        num_heads = self._infer_num_heads(hidden_size, cos_cache)
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} is not divisible by num_heads={num_heads}")
        head_dim = hidden_size // num_heads

        rotary_dim = self.rotary_embedding_dim if self.rotary_embedding_dim > 0 else min(head_dim, cos_cache.shape[-1] * 2)
        rotary_dim = min(rotary_dim, head_dim)
        rotary_dim -= rotary_dim % 2
        if rotary_dim == 0:
            return x

        x_heads = x.view(batch_size, seq_len, num_heads, head_dim)
        pos = position_ids.to(dtype=torch.long)
        cos = cos_cache.index_select(0, pos.reshape(-1)).view(*pos.shape, -1)
        sin = sin_cache.index_select(0, pos.reshape(-1)).view(*pos.shape, -1)

        half = rotary_dim // 2
        cos = cos[..., :half].to(dtype=x.dtype, device=x.device).unsqueeze(2)
        sin = sin[..., :half].to(dtype=x.dtype, device=x.device).unsqueeze(2)

        x_rot = x_heads[..., :rotary_dim]
        x_pass = x_heads[..., rotary_dim:]

        if self.interleaved:
            x_even = x_rot[..., 0::2]
            x_odd = x_rot[..., 1::2]
            rot_even = x_even * cos - x_odd * sin
            rot_odd = x_even * sin + x_odd * cos
            rotated = torch.stack((rot_even, rot_odd), dim=-1).flatten(-2)
        else:
            x_first = x_rot[..., :half]
            x_second = x_rot[..., half:rotary_dim]
            rot_first = x_first * cos - x_second * sin
            rot_second = x_first * sin + x_second * cos
            rotated = torch.cat((rot_first, rot_second), dim=-1)

        out = torch.cat((rotated, x_pass), dim=-1)
        return out.reshape(batch_size, seq_len, hidden_size)


class MultiHeadAttentionOp(nn.Module):
    def __init__(
        self,
        num_heads: int,
        scale: Optional[float],
        output_count: int = 3,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = scale
        self.output_count = output_count

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key: Optional[torch.Tensor] = None,
        past_value: Optional[torch.Tensor] = None,
    ):
        del bias, key_padding_mask, past_key, past_value

        batch_size, query_len, query_hidden = query.shape
        _, key_len, key_hidden = key.shape
        _, _, value_hidden = value.shape

        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be > 0, got {self.num_heads}")
        if query_hidden % self.num_heads != 0:
            raise ValueError(f"query_hidden={query_hidden} is not divisible by num_heads={self.num_heads}")
        if key_hidden % self.num_heads != 0 or value_hidden % self.num_heads != 0:
            raise ValueError("key/value hidden sizes are not divisible by num_heads")

        query_head_dim = query_hidden // self.num_heads
        key_head_dim = key_hidden // self.num_heads
        value_head_dim = value_hidden // self.num_heads

        query_heads = query.view(batch_size, query_len, self.num_heads, query_head_dim).transpose(1, 2)
        key_heads = key.view(batch_size, key_len, self.num_heads, key_head_dim).transpose(1, 2)
        value_heads = value.view(batch_size, key_len, self.num_heads, value_head_dim).transpose(1, 2)

        attn_scores = torch.matmul(query_heads, key_heads.transpose(-2, -1))
        scale = self.scale if (self.scale is not None and self.scale > 0) else (1.0 / math.sqrt(query_head_dim))
        attn_scores = attn_scores * scale

        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                attn_scores = attn_scores.masked_fill(~attention_mask, torch.finfo(attn_scores.dtype).min)
            else:
                attn_scores = attn_scores + attention_mask.to(dtype=attn_scores.dtype)

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attended = torch.matmul(attn_probs, value_heads)
        output = attended.transpose(1, 2).reshape(batch_size, query_len, value_hidden)

        if self.output_count == 1:
            return output

        outputs = [output]
        if self.output_count >= 2:
            outputs.append(key)
        if self.output_count >= 3:
            outputs.append(value)
        while len(outputs) < self.output_count:
            outputs.append(output.new_zeros(1))
        return tuple(outputs)


@add_converter(operation_type="SimplifiedLayerNormalization", version=14)
def _convert_simplified_layer_norm(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    del graph
    axis = int(node.attributes.get("axis", _AXIS_DEFAULT))
    epsilon = float(node.attributes.get("epsilon", _EPS_DEFAULT))
    module = SimplifiedLayerNormOp(axis=axis, epsilon=epsilon)
    return OperationConverterResult(torch_module=module, onnx_mapping=onnx_mapping_from_node(node))


@add_converter(operation_type="SkipSimplifiedLayerNormalization", version=1, domain="com.microsoft")
def _convert_skip_simplified_layer_norm(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    del graph
    epsilon = float(node.attributes.get("epsilon", _EPS_DEFAULT))
    module = SkipSimplifiedLayerNormOp(epsilon=epsilon, output_count=len(node.output_values))
    return OperationConverterResult(torch_module=module, onnx_mapping=onnx_mapping_from_node(node))


@add_converter(operation_type="RotaryEmbedding", version=1, domain="com.microsoft")
def _convert_rotary_embedding(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    del graph
    attrs = node.attributes
    module = RotaryEmbeddingOp(
        interleaved=int(attrs.get("interleaved", 0)),
        num_heads=int(attrs.get("num_heads", 0)),
        rotary_embedding_dim=int(attrs.get("rotary_embedding_dim", 0)),
    )
    return OperationConverterResult(torch_module=module, onnx_mapping=onnx_mapping_from_node(node))


@add_converter(operation_type="MultiHeadAttention", version=1, domain="com.microsoft")
def _convert_multi_head_attention(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    del graph
    attrs = node.attributes
    module = MultiHeadAttentionOp(
        num_heads=int(attrs.get("num_heads", 0)),
        scale=attrs.get("scale", None),
        output_count=len(node.output_values),
    )
    return OperationConverterResult(torch_module=module, onnx_mapping=onnx_mapping_from_node(node))
