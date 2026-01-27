import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import (
    apply_rotary_pos_emb,
    Qwen3Attention,
    repeat_kv,
)
from transformers.utils import logging

from baselines.snapkv.utils import init_snapkv

logger = logging.get_logger(__name__)

def qwen3_attn_forward_SnapKV(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    
    init_snapkv(self)

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        if hasattr(self, "kv_seq_len"): 
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    # Rotary embedding
    if position_embeddings is None:
       # For older transformers versions or if not passed
       # cos, sin = self.rotary_emb(value_states, position_ids)
       # Qwen3 usually passes position_embeddings. If not, we might be in trouble or need to infer.
       # But let's assume it's passed as per standard execution flow in this repo.
       pass 
    else:
       cos, sin = position_embeddings
       query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # [SnapKV] Strictly align with llama_model.py: Repeat KV unconditionally
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        
        # Check if prefill/prompt phase
        if key_states.shape[-2] == kv_seq_len:
            self.kv_seq_len = kv_seq_len 
            
            # SnapKV Compression Logic
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states, query_states, value_states, attention_mask, self.num_key_value_groups
            )
            
            past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)
        else:
            self.kv_seq_len += input_shape[1]
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        
        past_key_value._seen_tokens = self.kv_seq_len


    # Standard Attention Calculation
    # We use Strict Llama-aligned Attention Calculation without Chunking
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)


    if attn_output.size() != (input_shape[0], self.config.num_attention_heads, input_shape[1], self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(input_shape[0], self.config.num_attention_heads, input_shape[1], self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights
