import torch
from transformers import PreTrainedTokenizer
from typing import Optional

from .encoder_decoder_module import build_encoder_decoder_module
from .token_importance_extractor_module import build_token_importance_extractor_module

class Model(torch.nn.Module):
    """Class defining a model for the QaA generation task."""

    def __init__(self, model_name: str, tokenizer: Optional[PreTrainedTokenizer] = None, 
                 generation_params: Optional[dict] = None) -> None:
        """Create an instance of the model.

        Parameters
        ----------
        model_name : str
            The model name.
        tokenizer : (PreTrainedTokenizer, optional): 
            The tokenizer used by the model. Defaults to None.
        generation_params : dict, optional: 
            The parameters used in the gemeration procedure of the model. Defaults to None.
        """
        super().__init__()
        # Build the two model modules.
        self.token_importance_extractor_module = build_token_importance_extractor_module(model_name)
        self.encoder_decoder_module = build_encoder_decoder_module(model_name)

        # Add information about special tokens to the Encoder-Decoder module if the tokenizer is provided.
        if tokenizer is not None:
            self.encoder_decoder_module.config.decoder_start_token_id = tokenizer.cls_token_id
            self.encoder_decoder_module.config.pad_token_id = tokenizer.pad_token_id

        # Set generation parameters.
        if generation_params is None:
            self.generation_params = { 'do_sample': False, 'num_beams': 3, 'repetition_penalty': 2. }
        else:
            self.generation_params = generation_params

    def forward(self, input_ids: Optional[torch.LongTensor] = None, 
                attention_mask: Optional[torch.FloatTensor] = None,
                device: str = 'cuda',
                max_new_tokens: int = 50) -> torch.Tensor:
        """The forward function of the model.

        Parameters
        ----------
        input_ids : LongTensor, optional
            The input id tensor. Defaults to None.
        attention_mask : FloatTensor, optional
            The attention mask used to mask specific input ids. Defaults to None.
        device : str, optional: 
            The device used to perform the computations. Defaults to 'cuda'.
        max_new_tokens : int, optional
            Maximum number of tokens to use for the generated answer. Defaults to 50.
        
        Returns
        -------
        Tensor
            The tensor containing the generated answers.
        """
        output_1 = self.token_importance_extractor_module.to(device).forward(input_ids, attention_mask)

        return self.encoder_decoder_module.to(device).generate(input_ids, token_importances=output_1, 
                                                               max_new_tokens=max_new_tokens, **self.generation_params)
