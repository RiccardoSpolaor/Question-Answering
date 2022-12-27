import torch
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss
from typing import List, Optional, Tuple, Union
from transformers.utils import logging
from transformers.models.encoder_decoder.modeling_encoder_decoder import shift_tokens_right
from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
)



"""
These three functions overwrite the forward pass methods of the encoder-decoder distil roberta.
This code is heavily based on the huggin face code, taken from https://github.com/huggingface/transformers/blob/v4.25.1/src/transformers/models/roberta/modeling_roberta.py#L698

- The first function overwrites the forward pass of the encoder-decoder. Its only purpose is to take in input the token 
  importances scores and give them to the forward pass of the encoder.
- The second function overwrites the forward pass of the encoder. Its only purpose is to take in input the token importances
  scores and give them to the forward pass of the actual encoder.
- The thirs function overwrites the forward pass of the actual encoder. Its purpose is to finally use the token importances
  scores. For each encoder block, a linear layer is applied on the tokens importances scalar scores for generating vectors
  which are added to the block input vectors.

The exact same code of huggin face has been copy-pasted. The only modifications are enlighted through comments.
"""


def forward_encdec_overridden_roberta(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    token_importances: Optional[torch.LongTensor] = None,  # TOKENS IMPORTANCES
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
    """Overwrite the encoder-decoder forward pass"""

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

    kwargs_decoder = {
        argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
    }

    if encoder_outputs is None:
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            token_importances=token_importances,  # USAGE OF THE TOKENS IMPORTANCES IN THE ENCODER
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

    if (
        self.encoder.config.hidden_size != self.decoder.config.hidden_size
        and self.decoder.config.cross_attention_hidden_size is None
    ):
        encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

    if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
        decoder_input_ids = shift_tokens_right(
            labels, self.config.pad_token_id, self.config.decoder_start_token_id
        )

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

    loss = None
    if labels is not None:
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



def forward_enc_overridden_roberta(
    self,
    input_ids: Optional[torch.Tensor] = None,
    token_importances: Optional[torch.LongTensor] = None,  # TOKENS IMPORTANCES
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
    """Overwrite the encoder forward pass"""

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

    extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

    if self.config.is_decoder and encoder_hidden_states is not None:
        encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
        encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
    else:
        encoder_extended_attention_mask = None

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
        token_importances=token_importances,  # USAGE OF THE TOKENS IMPORTANCES IN THE ACTUAL ENCODER
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

def forward_encenc_overridden_roberta(
    self,
    hidden_states: torch.Tensor,
    token_importances: Optional[torch.LongTensor] = None,  # TOKENS IMPORTANCES
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
    """Overwrite the actual encoder forward pass"""

    all_hidden_states = () if output_hidden_states else None
    all_self_attentions = () if output_attentions else None
    all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

    next_decoder_cache = () if use_cache else None
    for i, layer_module in enumerate(self.layer):  # Iterate through each encoder block

        # USAGE OF THE TOKENS IMPORTANCES SCORES, BY APPLYING A LINEAR LAYER ON THEM AND ADD THE RESULTING VECTORS TO
        # THE INPUT VECTORS
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

