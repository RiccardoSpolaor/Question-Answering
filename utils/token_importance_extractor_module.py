import torch
from transformers import AutoModel

from typing import Optional


class _TokenImportancesExtractor(torch.nn.Module):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.decoder = AutoModel.from_pretrained(model_name)
        self.linear = torch.nn.Linear(self.decoder.config.hidden_size, 1)


    def forward(self, input_ids: Optional[torch.LongTensor] = None, 
                attention_mask: Optional[torch.FloatTensor] = None) -> torch.Tensor:
        decoder_outputs = self.decoder(input_ids=input_ids, attention_mask=attention_mask)

        decoder_hidden_states = decoder_outputs[0]

        logits = self.linear(decoder_hidden_states)

        return torch.sigmoid(logits)
    
def build_token_importance_extractor_module(model_name: str) -> _TokenImportancesExtractor:
    """Build the token importance extractor module, from the pretrained model.

    Parameters
    ----------
    model_name : str
        The model name from which the pretrained encoder is built.

    Returns
    -------
    PretrainedModel
        The token importance extractor module.
    """
    return _TokenImportancesExtractor(model_name)
