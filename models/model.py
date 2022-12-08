import torch

from transformers import EncoderDecoderModel, AutoTokenizer, PreTrainedTokenizer 

from .token_importances_extractor import TokenImportancesExtractor
from .encoder_decoder_distilroberta import build_encoder_decoder_distilroberta

from typing import Optional

class Model(torch.nn.Module):
    def __init__(self, model_name : str, tokenizer: Optional[PreTrainedTokenizer] = None, device : str = 'cpu'):
        super().__init__()
        self.model_name = model_name 
        self.device = device

        self.token_importances_extractor = TokenImportancesExtractor(model_name)
        self.token_importances_extractor.to(device)
        self.encoder_decoder = build_encoder_decoder_distilroberta()
        self.encoder_decoder.to(device)

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            # Add information about special tokens to the Encoder-Decoder module if the tokenizer is provided.
            self.tokenizer = tokenizer
            self.encoder_decoder_module.config.decoder_start_token_id = tokenizer.cls_token_id
            self.encoder_decoder_module.config.pad_token_id = tokenizer.pad_token_id


    def generate(self, passage : str, question : str, generation_params : dict = None):
        # Set generation parameters.
        if generation_params is None:
            self.generation_params = { 'do_sample': False, 'num_beams': 3, 'repetition_penalty': 2. }
        else:
            self.generation_params = generation_params

        inputs = self.tokenizer(
                question,
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

            generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return generated_text