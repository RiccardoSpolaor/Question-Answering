import math
import torch
import torch.nn as nn
from typing import Tuple, Optional

class LinearAttention(nn.Module):
    def __init__(self, old_attention_layer, k_dims=128, max_input_len=512):
        super().__init__()
        
        self.num_attention_heads = old_attention_layer.num_attention_heads
        self.attention_head_size = old_attention_layer.attention_head_size
        self.all_head_size       = old_attention_layer.all_head_size

        self.query = old_attention_layer.query
        self.key   = old_attention_layer.key
        self.value = old_attention_layer.value

        self.dropout = old_attention_layer.dropout
        self.position_embedding_type = old_attention_layer.position_embedding_type

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = old_attention_layer.max_position_embeddings
            self.distance_embedding      = old_attention_layer.distance_embedding

        self.is_decoder = old_attention_layer.is_decoder

        E = torch.randn(k_dims, max_input_len)/math.sqrt(k_dims)
        D = torch.randn(k_dims, max_input_len)/math.sqrt(k_dims)
        self.E = nn.Parameter(E)
        self.D = nn.Parameter(D) 

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        
        n_input = key_layer.shape[-2]

        if attention_mask is not None:
            if attention_mask.shape[-2]!=1:
                raise NotImplementedError(" Linformer not compatible with attention 2dim attention masks")
            else:
                n_input = torch.max(torch.sum(attention_mask==0,dim=(1,2,3)))

        projected_keys = torch.matmul(self.E[:,:n_input], key_layer[:,:,:n_input])
        projected_values = torch.matmul(self.D[:,:n_input], value_layer[:,:,:n_input])

        #attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        projected_attention_scores = torch.matmul(query_layer, projected_keys.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            raise NotImplementedError(" Linformer not compatible with relative keys")


        #attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        projected_attention_scores = projected_attention_scores / math.sqrt(self.attention_head_size)

        #print(torch.mean(projected_attention_scores),torch.var(projected_attention_scores),'s')


        # Normalize the attention scores to probabilities.
        #attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        projected_attention_probs = nn.functional.softmax(projected_attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        #attention_probs = self.dropout(attention_probs)
        projected_attention_probs = self.dropout(projected_attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            projected_attention_probs = projected_attention_probs * head_mask

        #context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = torch.matmul(projected_attention_probs, projected_values)


        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, projected_attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs