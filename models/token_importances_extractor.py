import torch
from typing import Optional
import torch.nn as nn
from transformers import AutoModel

from .linear_attention import LinearAttention


class TokenImportancesExtractor(nn.Module):

    def __init__(self, model_name : str, linear_attention=False, linearAttention_dims=128):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)

        if linear_attention:
            for L in self.encoder.encoder.layer:
                new_attention = LinearAttention(L.attention.self, k_dims=linearAttention_dims)
                L.attention.self = new_attention

        self.linear = nn.Linear(self.encoder.config.hidden_size, 1)


    def forward(self, input_ids: Optional[torch.LongTensor] = None, attention_mask: Optional[torch.FloatTensor] = None):

        encoder_outputs = self.encoder(
            input_ids = input_ids,
            attention_mask = attention_mask
        )

        encoder_hidden_states = encoder_outputs[0]

        logits = self.linear(encoder_hidden_states)

        return torch.sigmoid(logits) 
    