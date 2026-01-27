import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Union, Callable

from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3Model,
    Qwen3RMSNorm,
    Qwen3MLP,
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
    repeat_kv,
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import logging
from baselines.fastkv.utils import init_fastkv

logger = logging.get_logger(__name__)

class Qwen3FastKVAttention(Qwen3Attention):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        init_fastkv(self)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

            # [FastKV] Update KV with compression
            q_len = query_states.shape[2]
            if q_len > 1:
                key_states_compress, value_states_compress, self.tsp_idx = self.kv_cluster.update_kv(
                    key_states, query_states, value_states, attention_mask, self.num_key_value_groups, self.layer_idx)
                past_key_value.update(key_states_compress, value_states_compress, self.layer_idx, cache_kwargs)
            else:
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
                self.tsp_idx = None
        else:
            self.tsp_idx = None

        # Use SDPA instead of eager_attention_forward to save memory
        is_causal = query_states.shape[2] > 1
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=None,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal
        )
        attn_weights = None # We typically don't need weights if output_attentions=False
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


def qwen3_decoderlayer_forward_fastkv(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # [FastKV] Select important tokens and adjust position_ids at TSP layer
        tsp_idx = self.self_attn.tsp_idx
        if self.self_attn.kv_cluster.tsp_layer and tsp_idx is not None:
             # position_ids is not passed to decoder layer forward usually in qwen3 but it is in the signature?
             # Qwen3DecoderLayer.forward signature includes position_ids.
             if position_ids is not None:
                self.new_position_ids = torch.gather(position_ids, dim=1, index=tsp_idx)
             else:
                self.new_position_ids = None # Should not happen if we want to support this
            
             tsp_idx = tsp_idx.unsqueeze(-1)
             hidden_states = torch.gather(hidden_states, dim=1, index=tsp_idx.expand(-1, -1, hidden_states.size(2)))
        else:
            self.new_position_ids = None

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs

def qwen3_model_forward_fastkv(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs,
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    partial(decoder_layer.__call__, **flash_attn_kwargs),
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )
                
                # [FastKV] Update position_ids and position_embeddings
                new_position_ids = getattr(decoder_layer, "new_position_ids", None)
                if new_position_ids is not None:
                    position_ids = new_position_ids
                    # Recompute position embeddings with new position_ids
                    # Note: hidden_states in inputs_embeds for rotary_emb might be expected?
                    # Qwen3RotaryEmbedding usage: self.rotary_emb(value_states, position_ids) in attention
                    # But here in Model forward: self.rotary_emb(hidden_states, position_ids)
                    # The hidden_states here is what?
                    # "hidden_states = inputs_embeds" was passed initially.
                    # Inside the loop, it should be the current (compressed) hidden_states?
                    # The original code: position_embeddings = self.rotary_emb(hidden_states, position_ids)
                    # where hidden_states = inputs_embeds.
                    # So it uses input embeddings to determine dtype/device probably?
                    # Let's check layer_outputs[0], which is the new hidden_states.
                    position_embeddings = self.rotary_emb(layer_outputs[0], position_ids)
                    
                    # [FastKV] Update causal mask for compressed sequence
                    if causal_mask is not None:
                        new_len = layer_outputs[0].shape[1]
                        # Assuming causal_mask is (B, 1, Q, K) and Q=K
                        # We slice to (B, 1, new_len, new_len)
                        if causal_mask.shape[-1] >= new_len:
                             causal_mask = causal_mask[:, :, :new_len, :new_len]

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        # [FastKV] Cut-off hidden states like AdaKV
        # We need to return only the last token hidden state?
        # FastKV original llama_model.py: 
        # hidden_states = hidden_states[:, -1,:].unsqueeze(1)
        hidden_states = hidden_states[:, -1,:].unsqueeze(1)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
