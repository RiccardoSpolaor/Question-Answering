import torch 
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss
from transformers import EncoderDecoderModel
import types

import math

import warnings
from typing import List, Optional, Tuple, Union

from transformers.utils import logging
from transformers.models.encoder_decoder.modeling_encoder_decoder import DEPRECATION_WARNING, shift_tokens_right

from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
)



def build_encoder_decoder_distilroberta():
    model_name = 'distilroberta-base'
    encoder_decoder = EncoderDecoderModel.from_encoder_decoder_pretrained(model_name, model_name)

    # funcType = type(encoder_decoder.forward)
    encoder_decoder.forward = types.MethodType(_forward_encdec_overridden, encoder_decoder)

    # funcType = type(encoder_decoder.encoder.forward)
    encoder_decoder.encoder.forward = types.MethodType(_forward_enc_overridden, encoder_decoder.encoder)

    # funcType = type(encoder_decoder.encoder.encoder.forward)
    encoder_decoder.encoder.encoder.forward = types.MethodType(_forward_encenc_overridden, encoder_decoder.encoder.encoder)

    for L in encoder_decoder.encoder.encoder.layer:
        linear = nn.Linear(1, encoder_decoder.encoder.config.hidden_size)

        linear.weight = torch.nn.parameter.Parameter(torch.randn(linear.weight.shape, dtype=linear.weight.dtype)*1e-4)
        linear.bias = torch.nn.parameter.Parameter(torch.randn(linear.bias.shape, dtype=linear.bias.dtype)*1e-4)

        L.linear = linear 

        A1=LinAttention(L.attention.self, k_dims=64)
        L.attention.self=A1

    return encoder_decoder



def _forward_encdec_overridden(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    token_importances: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    decoder_input_ids: Optional[torch.LongTensor] = None,
    decoder_attention_mask: Optional[torch.BoolTensor] = None,
    encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
    past_key_values: Tuple[Tuple[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    **kwargs,
) -> Union[Tuple, Seq2SeqLMOutput]:
    r"""
    Returns:
    Examples:
    ```python
    >>> from transformers import EncoderDecoderModel, BertTokenizer
    >>> import torch
    >>> tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    >>> model = EncoderDecoderModel.from_encoder_decoder_pretrained(
    ...     "bert-base-uncased", "bert-base-uncased"
    ... )  # initialize Bert2Bert from pre-trained checkpoints
    >>> # training
    >>> model.config.decoder_start_token_id = tokenizer.cls_token_id
    >>> model.config.pad_token_id = tokenizer.pad_token_id
    >>> model.config.vocab_size = model.config.decoder.vocab_size
    >>> input_ids = tokenizer("This is a really long text", return_tensors="pt").input_ids
    >>> labels = tokenizer("This is the corresponding summary", return_tensors="pt").input_ids
    >>> outputs = model(input_ids=input_ids, labels=labels)
    >>> loss, logits = outputs.loss, outputs.logits
    >>> # save and load from pretrained
    >>> model.save_pretrained("bert2bert")
    >>> model = EncoderDecoderModel.from_pretrained("bert2bert")
    >>> # generation
    >>> generated = model.generate(input_ids)
    ```"""

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

    kwargs_decoder = {
        argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
    }

    if encoder_outputs is None:
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            token_importances=token_importances,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs_encoder,
        )
    elif isinstance(encoder_outputs, tuple):
        encoder_outputs = BaseModelOutput(*encoder_outputs)

    encoder_hidden_states = encoder_outputs[0]

    # optionally project encoder_hidden_states
    if (
        self.encoder.config.hidden_size != self.decoder.config.hidden_size
        and self.decoder.config.cross_attention_hidden_size is None
    ):
        encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

    if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
        decoder_input_ids = shift_tokens_right(
            labels, self.config.pad_token_id, self.config.decoder_start_token_id
        )

    # Decode
    decoder_outputs = self.decoder(
        input_ids=decoder_input_ids,
        attention_mask=decoder_attention_mask,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=attention_mask,
        inputs_embeds=decoder_inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        use_cache=use_cache,
        past_key_values=past_key_values,
        return_dict=return_dict,
        **kwargs_decoder,
    )

    # Compute loss independent from decoder (as some shift the logits inside them)
    loss = None
    if labels is not None:
        warnings.warn(DEPRECATION_WARNING, FutureWarning)
        logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.reshape(-1, self.decoder.config.vocab_size), labels.view(-1))

    if not return_dict:
        if loss is not None:
            return (loss,) + decoder_outputs + encoder_outputs
        else:
            return decoder_outputs + encoder_outputs

    return Seq2SeqLMOutput(
        loss=loss,
        logits=decoder_outputs.logits,
        past_key_values=decoder_outputs.past_key_values,
        decoder_hidden_states=decoder_outputs.hidden_states,
        decoder_attentions=decoder_outputs.attentions,
        cross_attentions=decoder_outputs.cross_attentions,
        encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        encoder_hidden_states=encoder_outputs.hidden_states,
        encoder_attentions=encoder_outputs.attentions,
    )



def _forward_enc_overridden(
    self,
    input_ids: Optional[torch.Tensor] = None,
    token_importances: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    token_type_ids: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
    r"""
    encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
        Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
        the model is configured as a decoder.
    encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
        the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
        - 1 for tokens that are **not masked**,
        - 0 for tokens that are **masked**.
    past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
        Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
        If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
        don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
        `decoder_input_ids` of shape `(batch_size, sequence_length)`.
    use_cache (`bool`, *optional*):
        If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
        `past_key_values`).
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if self.config.is_decoder:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
    else:
        use_cache = False

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        input_shape = input_ids.size()
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    batch_size, seq_length = input_shape
    device = input_ids.device if input_ids is not None else inputs_embeds.device

    # past_key_values_length
    past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

    if attention_mask is None:
        attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

    if token_type_ids is None:
        if hasattr(self.embeddings, "token_type_ids"):
            buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
            buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
            token_type_ids = buffered_token_type_ids_expanded
        else:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

    # If a 2D or 3D attention mask is provided for the cross-attention
    # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
    if self.config.is_decoder and encoder_hidden_states is not None:
        encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
        encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
    else:
        encoder_extended_attention_mask = None

    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head
    # attention_probs has shape bsz x n_heads x N x N
    # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
    # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
    head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

    embedding_output = self.embeddings(
        input_ids=input_ids,
        position_ids=position_ids,
        token_type_ids=token_type_ids,
        inputs_embeds=inputs_embeds,
        past_key_values_length=past_key_values_length,
    )
    
    encoder_outputs = self.encoder(
        embedding_output,
        token_importances=token_importances,
        attention_mask=extended_attention_mask,
        head_mask=head_mask,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_extended_attention_mask,
        past_key_values=past_key_values,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    sequence_output = encoder_outputs[0]
    pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

    if not return_dict:
        return (sequence_output, pooled_output) + encoder_outputs[1:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        last_hidden_state=sequence_output,
        pooler_output=pooled_output,
        past_key_values=encoder_outputs.past_key_values,
        hidden_states=encoder_outputs.hidden_states,
        attentions=encoder_outputs.attentions,
        cross_attentions=encoder_outputs.cross_attentions,
    )



logger = logging.get_logger(__name__)

def _forward_encenc_overridden(
    self,
    hidden_states: torch.Tensor,
    token_importances: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.FloatTensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = False,
    output_hidden_states: Optional[bool] = False,
    return_dict: Optional[bool] = True,
) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
    all_hidden_states = () if output_hidden_states else None
    all_self_attentions = () if output_attentions else None
    all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

    next_decoder_cache = () if use_cache else None
    for i, layer_module in enumerate(self.layer):

        hidden_states = hidden_states + layer_module.linear( token_importances )

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        layer_head_mask = head_mask[i] if head_mask is not None else None
        past_key_value = past_key_values[i] if past_key_values is not None else None

        if self.gradient_checkpointing and self.training:

            if use_cache:
                logger.warning(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs, past_key_value, output_attentions)

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(layer_module),
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
            )
        else:
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )

        hidden_states = layer_outputs[0]
        if use_cache:
            next_decoder_cache += (layer_outputs[-1],)
        if output_attentions:
            all_self_attentions = all_self_attentions + (layer_outputs[1],)
            if self.config.add_cross_attention:
                all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
        return tuple(
            v
            for v in [
                hidden_states,
                next_decoder_cache,
                all_hidden_states,
                all_self_attentions,
                all_cross_attentions,
            ]
            if v is not None
        )
    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=next_decoder_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attentions,
        cross_attentions=all_cross_attentions,
    )



class LinAttention(nn.Module):
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

        if torch.any(torch.isnan(hidden_states)):
            print('0')

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

        if torch.any(torch.isnan(key_layer)):
            print('1_1')
        if torch.any(torch.isnan(value_layer)):
            print('1_2')
        if torch.any(torch.isnan(query_layer)):
            print('1_3')

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

        #print(torch.mean(key_layer),torch.var(key_layer),'k')
        #print(torch.mean(value_layer),torch.var(value_layer),'v')

        projected_keys = torch.matmul(self.E[:,:n_input], key_layer[:,:,:n_input])
        projected_values = torch.matmul(self.D[:,:n_input], value_layer[:,:,:n_input])

        #print(torch.mean(self.E),torch.var(self.E),'E')

        #print(torch.mean(projected_keys),torch.var(projected_keys),'pk')
        #print(torch.mean(projected_values),torch.var(projected_values),'pv')

        if torch.any(torch.isnan(projected_keys)):
            print('2_1')
        if torch.any(torch.isnan(projected_values)):
            print('2_2')

        #print(torch.mean(query_layer),torch.var(query_layer),'q')

        #attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        projected_attention_scores = torch.matmul(query_layer, projected_keys.transpose(-1, -2))

        if torch.any(torch.isnan(projected_attention_scores)):
            print('3_1')

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            raise NotImplementedError(" Linformer not compatible with relative keys")


        #attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        projected_attention_scores = projected_attention_scores / math.sqrt(self.attention_head_size)

        #print(torch.mean(projected_attention_scores),torch.var(projected_attention_scores),'s')

        if torch.any(torch.isnan(projected_attention_scores)):
            print('3_2')

        # Normalize the attention scores to probabilities.
        #attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        projected_attention_probs = nn.functional.softmax(projected_attention_scores, dim=-1)

        if torch.any(torch.isnan(projected_attention_probs)):
            print('4_1')
            print(projected_attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        #attention_probs = self.dropout(attention_probs)
        projected_attention_probs = self.dropout(projected_attention_probs)

        if torch.any(torch.isnan(projected_attention_probs)):
            print('4_2')

        # Mask heads if we want to
        if head_mask is not None:
            projected_attention_probs = projected_attention_probs * head_mask

        if torch.any(torch.isnan(projected_attention_probs)):
            print('4_3')

        #context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = torch.matmul(projected_attention_probs, projected_values)

        #print(torch.mean(context_layer),torch.var(context_layer),'C')

        if torch.any(torch.isnan(context_layer)):
            print('5_1')

        #print(projected_attention_probs)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, projected_attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs