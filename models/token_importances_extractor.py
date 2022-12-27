import torch
from typing import Optional
import torch.nn as nn
from transformers import AutoModel

from .linear_attention import LinearAttention


class TokenImportancesExtractor(nn.Module):
    """Tokens importances extractor module.

    It takes in input the question, the passage and, optionally, the history, and it computes the importance score of each 
    input token. This importance score is in [0,1] and it represents the likelihood of the token to be in the passage span 
    containing the answer.

    This module is a transformer-based encoder (e.g. bert) with a linear layer on top.
    Basically, for each input token, a contextual embedding vector is produced using the encoder, and then a probability score
    is computed using the linear layer.

    This module is built from a pre-trained transformer-based encoder: either bert-tiny ('prajjwal1/bert-tiny') or distil 
    roberta ('distilroberta-base').

    Parameters
    ----------
    model_name : str
        Name of the pre-trained model to use, either 'prajjwal1/bert-tiny' or 'distilroberta-base'.
    linear_attention : bool, optional
        Whether to use the linear attention or not, by default False.
        The linear attention is implemented using the Linformer.
    linearAttention_dims : int, optional
        Dimensionality of the linear attention, by default 128
    """

    def __init__(self, model_name : str, linear_attention=False, linearAttention_dims=128):
        super().__init__()

        # Pre-trained encoder
        self.encoder = AutoModel.from_pretrained(model_name)

        if linear_attention:
            for L in self.encoder.encoder.layer:
                new_attention = LinearAttention(L.attention.self, k_dims=linearAttention_dims)
                L.attention.self = new_attention

        # Linear layer on top of the encoder, producing the scalar scores
        self.linear = nn.Linear(in_features=self.encoder.config.hidden_size, out_features=1)


    def forward(self, input_ids: Optional[torch.LongTensor] = None, attention_mask: Optional[torch.FloatTensor] = None):
        """Forward pass

        Parameters
        ----------
        input_ids : Optional[torch.LongTensor], optional
            Input tokens, by default None
        attention_mask : Optional[torch.FloatTensor], optional
            Input tokens attention mask, by default None

        Returns
        -------
        Tensor
            Tokens importances scores
        """

        # Encoder forward pass
        encoder_outputs = self.encoder(
            input_ids = input_ids,
            attention_mask = attention_mask
        )

        encoder_hidden_states = encoder_outputs[0]

        # Linear layer forward pass
        logits = self.linear(encoder_hidden_states)

        return torch.sigmoid(logits) 
    