from transformers import EncoderDecoderModel
import types
import torch 
import torch.nn as nn

from .encoder_decoder_distilroberta import _forward_encdec_overridden_roberta, _forward_encenc_overridden_roberta, _forward_enc_overridden_roberta
from .encoder_decoder_bertTiny import _forward_encdec_overridden_bertTiny, _forward_encenc_overridden_bertTiny, _forward_enc_overridden_bertTiny

from .linear_attention import LinearAttention

def build_encoder_decoder(model_name, linear_attention=False, linearAttention_dims=128):
    encoder_decoder = EncoderDecoderModel.from_encoder_decoder_pretrained(model_name, model_name)

    if model_name=='distilroberta-base':
        _forward_encdec_overridden = _forward_encdec_overridden_roberta
        _forward_encenc_overridden = _forward_encenc_overridden_roberta
        _forward_enc_overridden = _forward_enc_overridden_roberta
    elif model_name=='prajjwal1/bert-tiny':
        _forward_encdec_overridden = _forward_encdec_overridden_bertTiny
        _forward_encenc_overridden = _forward_encenc_overridden_bertTiny
        _forward_enc_overridden = _forward_enc_overridden_bertTiny
    else:
        raise ValueError('`model_name` must be either "distilroberta-base" or "prajjwal1/bert-tiny"')

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

        if linear_attention:
            new_attention = LinearAttention(L.attention.self, k_dims=linearAttention_dims)
            L.attention.self = new_attention

    return encoder_decoder