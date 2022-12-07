import torch
from typing import Optional

from .encoder_decoder_module import build_encoder_decoder_module
from .token_importance_extractor_module import build_token_importance_extractor_module

class Model(torch.nn.Module):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.token_importance_extractor_module = build_token_importance_extractor_module(model_name)
        self.encoder_decoder_module = build_encoder_decoder_module(model_name)

    def forward(self, input_ids: Optional[torch.LongTensor] = None, 
                attention_mask: Optional[torch.FloatTensor] = None,
                device: str = 'cuda') -> torch.Tensor:
            output_1 = self.token_importance_extractor_module.to(device).forward(input_ids, attention_mask)

            return self.encoder_decoder_module.to(device).generate(input_ids, token_importances=output_1, max_new_tokens=50)
    