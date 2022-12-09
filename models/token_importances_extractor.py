import torch

from typing import Optional
import torch.nn as nn
from transformers import AutoModel


class TokenImportancesExtractor(nn.Module):

    def __init__(self, model_name : str):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)

        self.linear = nn.Linear(self.encoder.config.hidden_size, 1)


    def forward(self, input_ids: Optional[torch.LongTensor] = None, attention_mask: Optional[torch.FloatTensor] = None):

        encoder_outputs = self.encoder(
            input_ids = input_ids,
            attention_mask = attention_mask
        )

        encoder_hidden_states = encoder_outputs[0]

        logits = self.linear(encoder_hidden_states)

        return torch.sigmoid(logits) 
    