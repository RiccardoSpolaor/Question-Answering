from transformers import EncoderDecoderModel
import types
import torch 
import torch.nn as nn

from .encoder_decoder_distilroberta import (forward_encdec_overridden_roberta, forward_encenc_overridden_roberta, 
                                            forward_enc_overridden_roberta)
from .encoder_decoder_bertTiny import (forward_encdec_overridden_bertTiny, forward_encenc_overridden_bertTiny, 
                                       forward_enc_overridden_bertTiny)



def build_encoder_decoder(model_name):
    """Return the encoder-decoder module, i.e. the seq2seq module.

    Given the question, the passage (and, optionally, the history) and the tokens importances scores, it generates the answer.

    It is a classic transformer-based encoder-decoder modified to make it accept the tokens importances as second input of 
    the encoder. The importances are injected inside the model by combining them to the input hidden states of every encoder
    block. More precisely, for each encoder block, the tokens importances scores are transformed into vectors of the same 
    dimensionality of the block inputs vectors using a linear layer: then, these tokens importances vectors are simply added 
    to the block inputs vectors. We have chosen to use a different linear layer for every block of the encoder: therefore 
    there $n$ additional linear layers, where $n$ is the number of encoder blocks.

    This module is built from a pre-trained transformer-based encoder-decoder: either bert-tiny ('prajjwal1/bert-tiny') or 
    distil roberta ('distilroberta-base').

    The implementation of this module is heavily based on the implementation of the corresponding pre-trained model from 
    hugging face.
    Basically, the pre-trained model is loaded (either bert tiny or distil roberta), and the forward methods are overwritten
    in order to pass and use the tokens importances scores.
    In particular, the following methods are overwritten.
    - The forward method of the encoder-decoder, in order to take in input the tokens importances scores and give them in 
      input to the forward pass of the encoder.
    - The forward method of the encoder, in order to take in input the tokens importances scores and give them in input to 
      the forward pass of the actual encoder.
    - The forward method of the actual encoder, in order to take in input the tokens importances scores and use them in each 
      encoder block, by means of an additional linear layer.

    Parameters
    ----------
    model_name : str
        Name of the pre-trained model to use, either 'prajjwal1/bert-tiny' or 'distilroberta-base'.

    Returns
    -------
    EncoderDecoderModel
        The encoder-decoder module

    Raises
    ------
    ValueError
        If the specified `model_name` is neither "distilroberta-base" nor "prajjwal1/bert-tiny".
    """

    # Pre-trained encoder-decoder
    encoder_decoder = EncoderDecoderModel.from_encoder_decoder_pretrained(model_name, model_name)

    # Overwrite the forward methods
    if model_name=='distilroberta-base':
        # Overwrite the encoder-decoder forward pass
        _forward_encdec_overridden = forward_encdec_overridden_roberta
        # Overwrite the encoder forward pass
        _forward_encenc_overridden = forward_encenc_overridden_roberta
        # Overwrite the actual encoder forward pass
        _forward_enc_overridden = forward_enc_overridden_roberta
    elif model_name=='prajjwal1/bert-tiny':
        # Overwrite the encoder-decoder forward pass
        _forward_encdec_overridden = forward_encdec_overridden_bertTiny
        # Overwrite the encoder forward pass
        _forward_encenc_overridden = forward_encenc_overridden_bertTiny
        # Overwrite the actual encoder forward pass
        _forward_enc_overridden = forward_enc_overridden_bertTiny
    else:
        raise ValueError('`model_name` must be either "distilroberta-base" or "prajjwal1/bert-tiny"')

    encoder_decoder.forward = types.MethodType(_forward_encdec_overridden, encoder_decoder)
    encoder_decoder.encoder.forward = types.MethodType(_forward_enc_overridden, encoder_decoder.encoder)
    encoder_decoder.encoder.encoder.forward = types.MethodType(_forward_encenc_overridden, encoder_decoder.encoder.encoder)

    # Add a new linear layer for each encoder block.
    # Iterate through each encoder block.
    for L in encoder_decoder.encoder.encoder.layer:
        # New linear layer, taking in input the tokens importances scala scores and producing vectors of the same dimension
        # of the inputs
        linear = nn.Linear(in_features=1, out_features=encoder_decoder.encoder.config.hidden_size)

        # Inizialization of the new linear layer with random weigths
        linear.weight = torch.nn.parameter.Parameter(torch.randn(linear.weight.shape, dtype=linear.weight.dtype)*1e-4)
        linear.bias = torch.nn.parameter.Parameter(torch.randn(linear.bias.shape, dtype=linear.bias.dtype)*1e-4)

        # Inject this new linear layer into the current block
        L.linear = linear 

    return encoder_decoder