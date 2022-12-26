from IPython.display import display
import numpy as np
import os
import time
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from typing import Optional

from .checkpoint_loader import get_checkpoint_path
from models.model import Model


def _loss_function_token_importances_extractor(probabilities: Tensor, target: Tensor) -> Tensor:
    """Get the loss score of the predictions of Token Importances Extractor module.

    Parameters
    ----------
    probabilities : Tensor
        The token importances predicted probabilities.
    target : Tensor
        The actual token importances.

    Returns
    -------
    Tensor
        The loss score of the predictions of Token Importances Extractor module.
    """
    loss = - torch.log(probabilities) * (target) / (torch.sum(target, 1, keepdim=True) + 1)
    loss += - torch.log(1 - probabilities) * (1 - target) / (torch.sum(1 - target, 1, keepdim=True) + 1)
    return torch.mean(torch.sum(loss, (1, 2)))

def train(train_dataloader: DataLoader, val_dataloader: DataLoader, model: Model, use_history: bool,
          checkpoint_path: Optional[str] = None, epochs: int = 3, learning_rate: float = 5e-5, 
          optimizer_state_dict: Optional[dict] = None, steps_per_update: int = 1, steps_empty_cache: Optional[int] = None,
          steps_validate: Optional[int] = None, steps_save: Optional[int] = None, loss_history: Optional[list] = None, 
          val_loss_history: Optional[list] = None, seed=None, device: str = 'cpu') -> None:

    # Create path to save checkpoints
    if checkpoint_path is None:
        checkpoint_path = get_checkpoint_path(model.model_name, seed, use_history)

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    if loss_history is None:
        loss_history=[]
        val_loss_history=[]

    token_importances_extractor = model.token_importances_extractor
    encoder_decoder = model.encoder_decoder
    tokenizer = model.tokenizer

    optimizer = torch.optim.Adam(iter(list(model.parameters())), lr=learning_rate)

    # Set the state of the optimizer if the training is restarted from a checkpoint
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)

    tot_steps = len(train_dataloader) * epochs
    n_steps = 0
    skip_steps = len(loss_history)

    for epoch in range(epochs):
        # Set up display element
        disp = display('', display_id=True)

        torch.cuda.empty_cache() # Remove unused tensors from gpu memory
        # Initialize running losses
        running_loss1 = 0.0
        running_loss2 = 0.0
        optimizer.zero_grad()
        start_time = time.time()
        batch_steps=0
        for batch_idx, data in enumerate(train_dataloader, 0):
            # Necessary to restart the dataloader from the checkpoint mantaining the seed
            if n_steps < skip_steps:
                n_steps += 1
                continue
            n_steps += 1
            batch_steps += 1

            # Get the data
            (passage, question, history), (answer, sep_starts, sep_ends) = data
            history = tuple([h.split(' <sep> ') for h in history])

            if use_history:
                separator = f' {tokenizer.sep_token} '
                question = tuple([q + f'{separator if len(h) else ""}' + separator.join(h) for q, h in zip(question, history)])

            # Tokenize inputs and targets
            inputs = tokenizer(
                question,
                passage,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(device)

            targets = tokenizer(
                list(answer),
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(device)

            # Compute targets for token importance extractor
            y = torch.zeros(inputs.input_ids.shape+(1,), device=device)
            for i in range(len(sep_starts)):
                start_tok = inputs.char_to_token(i,sep_starts[i],1)
                end_tok = inputs.char_to_token(i,sep_ends[i],1)
                if start_tok is None:
                    start_tok = inputs.char_to_token(i,sep_starts[i]+1,1)
                if start_tok is None:
                    start_tok = inputs.char_to_token(i,sep_starts[i]-1,1)
                if end_tok is None:
                    end_tok = inputs.char_to_token(i,sep_ends[i]-1,1)
                if end_tok is None:
                    end_tok = inputs.char_to_token(i,sep_ends[i]+1,1)
                y[i, start_tok : end_tok] = 1 

            # Compute importances
            out1 = token_importances_extractor.forward(inputs.input_ids,
                            inputs.attention_mask)

            # Calculate factor for teacher forcing
            forcing = 0.5 + np.cos(np.pi*(epoch*len(train_dataloader)+batch_idx)/tot_steps)/2
            importances = forcing * y + (1-forcing) * out1

            # Predict answer
            out2 = encoder_decoder(input_ids = inputs.input_ids,
                         labels = targets.input_ids,
                         token_importances = importances)

            # Obtain loss
            loss1 = _loss_function_token_importances_extractor(out1, y)
            loss2 = out2.loss
            loss = loss1 + loss2

            loss.backward()

            # Update
            if batch_idx % steps_per_update == steps_per_update-1:
                if isinstance(steps_per_update, float):
                    steps_per_update=int(steps_per_update*len(train_dataloader))
                optimizer.step()
                optimizer.zero_grad()

            # Clear memory
            if steps_empty_cache is not None:
                if isinstance(steps_empty_cache, float):
                    steps_empty_cache=int(steps_empty_cache*len(train_dataloader))
                if batch_idx % steps_empty_cache == steps_empty_cache-1:
                    torch.cuda.empty_cache()

            # Evaluate on validation set
            if steps_validate is not None:
                if isinstance(steps_validate, float):
                    steps_validate=int(steps_validate*len(train_dataloader))
                if batch_idx % steps_validate == steps_validate-1:
                    torch.cuda.empty_cache()

                    val_l1, val_l2 = _loss_validate( tokenizer, token_importances_extractor, encoder_decoder, val_dataloader, disp,
                                                    use_history=use_history, device=device)

                    val_loss_history.append([len(loss_history), val_l1.detach().cpu().numpy(), val_l2.detach().cpu().numpy()])
                    torch.cuda.empty_cache()

            # Update history and print                    
            running_loss1 += loss1.item()
            running_loss2 += loss2.item()
            loss_history.append([loss1.detach().cpu().numpy(), loss2.detach().cpu().numpy()])
            epoch_time = time.time() - start_time
            batch_time = epoch_time/(batch_idx+1)
            disp.update(f"epoch: {epoch + 1}/{epochs}, {batch_idx + 1}/{len(train_dataloader)}, {epoch_time:.0f}s {batch_time*1e3:.0f}ms/step, lr: {optimizer.param_groups[0]['lr']:.3g}, forcing={forcing:.3g}, loss: {running_loss1/batch_steps:.3g} {running_loss2/batch_steps:.3g}")

            # Save checkpoints
            if steps_save is not None:
                if isinstance(steps_save, float):
                    steps_save=int(steps_save*len(train_dataloader))
                if batch_idx % steps_save == steps_save-1:
                    torch.cuda.empty_cache()

                    checkpoint = {'model_state_dict': model.state_dict(),
                                  'opt_state_dict' : optimizer.state_dict(),
                                  'loss_history': np.array(loss_history),
                                  'val_loss_history': np.array(val_loss_history)}
                    torch.save(checkpoint, checkpoint_path)
                    torch.cuda.empty_cache()

    # Final checkpoint save
    torch.cuda.empty_cache()

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'opt_state_dict' : optimizer.state_dict(),
        'loss_history': np.array(loss_history),
        'val_loss_history': np.array(val_loss_history)
        }

    torch.save(checkpoint, checkpoint_path)

    torch.cuda.empty_cache()

def _loss_validate(tokenizer, token_importances_extractor, encoder_decoder, val_dataloader, disp,
                   use_history = False, device='cpu'):
    tot_l1=0
    tot_l2=0
    n=0
    t0=time.time()

    torch.cuda.empty_cache()

    for batch_idx, data in enumerate(val_dataloader, 0):
        with torch.no_grad():
            # get the inputs; data is a list of [inputs, labels]
            (passage, question, history), (answer, sep_starts, sep_ends) = data
            history = tuple([h.split(' <sep> ') for h in history])

            if use_history:
                separator = f' {tokenizer.sep_token} '
                question = tuple([q + f'{separator if len(h) else ""}' + separator.join(h) for q, h in zip(question, history)])

            inputs = tokenizer(
                question,
                passage,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(device)

            labels = tokenizer(
                list(answer),
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(device)

            out1 = token_importances_extractor.forward( inputs.input_ids,
                                                        inputs.attention_mask)

            y = torch.zeros(inputs.input_ids.shape+(1,), device=out1.device)

            for i in range(len(sep_starts)):

                start_tok = inputs.char_to_token(i,sep_starts[i],1)
                end_tok = inputs.char_to_token(i,sep_ends[i],1)

                if start_tok is None:
                    start_tok = inputs.char_to_token(i,sep_starts[i]+1,1)
                if start_tok is None:
                    start_tok = inputs.char_to_token(i,sep_starts[i]-1,1)

                if end_tok is None:
                    end_tok = inputs.char_to_token(i,sep_ends[i]-1,1)
                if end_tok is None:
                    end_tok = inputs.char_to_token(i,sep_ends[i]+1,1)

                y[i, start_tok : end_tok] = 1

            loss1 = _loss_function_token_importances_extractor(out1,y)

            out2 = encoder_decoder( input_ids = inputs.input_ids,
                                    attention_mask = inputs.attention_mask,
                                    labels = labels.input_ids,
                                    token_importances = out1)

            loss2 = out2.loss

            l=(len(question) if isinstance(question,tuple) else 1)
            tot_l1 += loss1*l
            tot_l2 += loss2*l
            n += l

        disp.update(f"validate {batch_idx + 1}/{len(val_dataloader)}, {(time.time()-t0):.0f}s {(time.time()-t0) / (batch_idx+1)*1e3:.0f}ms/step, mean losses: {tot_l1/n:.3g} {tot_l2/n:.3g}")

    return tot_l1 / n, tot_l2 / n
