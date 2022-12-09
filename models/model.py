import torch

from transformers import AutoTokenizer, PreTrainedTokenizer 

from .token_importances_extractor import TokenImportancesExtractor
from .encoder_decoder import build_encoder_decoder

from typing import List, Optional

class Model(torch.nn.Module):
    def __init__(self, model_name : str, tokenizer: Optional[PreTrainedTokenizer] = None, linear_attention=False, 
                linearAttention_dims=128, device : str = 'cpu'):
        super().__init__()
        self.model_name = model_name 
        self.device = device

        self.token_importances_extractor = TokenImportancesExtractor(model_name)
        self.token_importances_extractor.to(device)
        self.encoder_decoder = build_encoder_decoder(model_name=model_name, linear_attention=linear_attention,
                                                     linearAttention_dims=linearAttention_dims)
        self.encoder_decoder.to(device)

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            # Add information about special tokens to the Encoder-Decoder module if the tokenizer is provided.
            self.tokenizer = tokenizer
        self.encoder_decoder.config.decoder_start_token_id = self.tokenizer.cls_token_id
        self.encoder_decoder.config.pad_token_id = self.tokenizer.pad_token_id


    def generate(self, passage : str, question : str, history : Optional[List[str]] = None, 
                 generation_params : Optional[dict] = None) -> str:
        # Set generation parameters.
        if generation_params is None:
            self.generation_params = { 'do_sample': False, 'num_beams': 3, 'repetition_penalty': 2. }
        else:
            self.generation_params = generation_params

        # Add history to question if present
        if history is not None:
            question_and_history=[q + self.tokenizer.sep_token + h for q, h in zip(question, history)]
        else:
            question_and_history = question


        inputs = self.tokenizer(
                question_and_history,
                passage,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

        if generation_params is None:
            generation_params = {
                'do_sample' : False,
                'num_beams' : 3,
                'repetition_penalty' : 2.
            }

        with torch.no_grad():
            token_importances_output = self.token_importances_extractor.forward(inputs.input_ids, inputs.attention_mask)

            generated_ids = self.encoder_decoder.generate(inputs.input_ids, token_importances=token_importances_output, 
                                                          **generation_params)

            generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return generated_text