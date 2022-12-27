from IPython.display import display, DisplayHandle
import numpy as np
import os
import time
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer
from models.token_importances_extractor import TokenImportancesExtractor
from transformers import EncoderDecoderModel
from typing import Optional

from .checkpoint_loader import get_checkpoint_path
from models.model import Model



def train(train_dataloader: DataLoader, val_dataloader: DataLoader, model: Model, use_history: bool,
          checkpoint_path: Optional[str] = None, epochs: int = 3, learning_rate: float = 5e-5, 
          optimizer_state_dict: Optional[dict] = None, steps_per_update: int = 1, steps_empty_cache: Optional[int] = None,
          steps_validate: Optional[int] = None, steps_save: Optional[int] = None, loss_history: Optional[list] = None, 
          val_loss_history: Optional[list] = None, seed: int = None, device: str = 'cuda') -> None:
    """Train the given model on the given training dataset.

    We remind that the model consists of two parts: a tokens importances extractor and an encoder-decoder (i.e. seq2seq).
    Accordingly, the loss function is the sum of two losses.
    - The first loss is about the goodness of the tokens importances extractor. Basically, it measures the difference 
      between the true span in the passage and the importances over the passage assigned by the extractor.
      In particular, this loss consists in computing the binary cross-entropy, where: the true label of each passage token
      is 1 if that token is in the span, 0 otherwise; the predicted probability is the one assigned by the extractor.
      More precisely, a variant of the binary crossentropy for imbalance classes is used, since there are much more 0 than
      1.
      See `_loss_function_token_importances_extractor`
    - The second loss is about the goodness of the encoder-decoder. Basically, it measures the difference between the 
      generated answer and the true one. This is measured using the same standard loss function of the encoder-decoder, which
      is the crossentropy loss.

    For each training step, the loss is computed by adding these two losses on the current batch. Then, the backpropagation 
    using this loss is performed on all the parameters of the model.   

    Parameters
    ----------
    train_dataloader : DataLoader
        Dataloader used for training.
    val_dataloader : DataLoader
        Dataloader used for validation.
    model : Model
        Model to train.
    use_history : bool
        Whether to use the history or not.
        If True, the history is given as additional input to the model.
    checkpoint_path : Optional[str], optional
        Path into which storing the training checkpoint, by default None
    epochs : int, optional
        Number of training epochs, by default 3
    learning_rate : float, optional
        Learning rate, by default 5e-5
    optimizer_state_dict : Optional[dict], optional
        Dictionary containing the state of the optimizer, by default None.
        This is given if a previous training has already been done, and we want to start from it (i.e. incremental training).
    steps_per_update : int, optional
        Number of training steps for updating the parameters of the model, by default 1.
        Basically, the losses computed on the training steps are accumulated for `steps_per_update` steps: then, the 
        backpropagation is performed. 
        This can be useful because a small batch size is used (e.g. 8): so, accumulating the losses before each update can 
        be a good idea in order to have better changes on the parameters.
    steps_empty_cache : Optional[int], optional
        Number of steps for emptying the cache, by default None. 
        This is done for erasing the cuda memory. 
        If None, the cache is never emptied (only at the beginning of each epoch).
    steps_validate : Optional[int], optional
        Number of steps for computing the validation loss, by default None.
        Basically, the validation loss is computed only after `steps_validate` steps.
        If None, the validation loss is not computed.
    steps_save : Optional[int], optional
        Number of steps for saving the training checkpoint, by default None.
        A training checkpoint consists of the model weigths, the optimizer state, the training loss history and the validation
        loss history. 
        If None, the training checkpoint is never saved.
    loss_history : Optional[list], optional
        Training loss history, by default None.
        This is given if a previous training has already been done, and we want to start from it (i.e. incremental training).
    val_loss_history : Optional[list], optional
        Validation loss history, by default None.
        This is given if a previous training has already been done, and we want to start from it (i.e. incremental training).
    seed : int, optional
        Random seed, by default None
    device : str, optional
        Device onto which attach the training, by default 'cuda'
    """

    # Create path to save checkpoints
    if checkpoint_path is None:
        checkpoint_path = get_checkpoint_path(model.model_name, seed, use_history)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # No previous training: new loss histories
    if loss_history is None:
        loss_history = []
        val_loss_history = []

    # Different parts of the model
    token_importances_extractor = model.token_importances_extractor
    encoder_decoder = model.encoder_decoder
    tokenizer = model.tokenizer

    # Adam optimizer, with the default learning rate
    optimizer = torch.optim.Adam(iter(list(model.parameters())), lr=learning_rate)

    # Set the state of the optimizer if the training is restarted from a checkpoint
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)

    # Total steps to perform
    tot_steps = len(train_dataloader) * epochs
    # Number of step already done
    n_steps = 0
    # Steps to skip (because incremental solving from a previous checkpoint)
    skip_steps = len(loss_history)

    # Iterate across the epochs
    for epoch in range(epochs):
        # Set up display element
        disp = display('', display_id=True)

        torch.cuda.empty_cache() # Remove unused tensors from gpu memory

        # Initialize running losses
        running_loss1 = 0.0
        running_loss2 = 0.0
        optimizer.zero_grad()

        start_time = time.time()

        # Number of batches for the current update step
        batch_steps=0

        for batch_idx, data in enumerate(train_dataloader, 0):
            # Necessary to restart the dataloader from the checkpoint mantaining the seed
            if n_steps < skip_steps:
                n_steps += 1
                continue

            # Increment the number of steps already done and the number of batches in the current update step
            n_steps += 1
            batch_steps += 1

            # Get the data
            (passage, question, history), (answer, sep_starts, sep_ends) = data
            history = tuple([h.split(' <sep> ') for h in history])

            # Inject the history into the input
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

            # Compute targets for token importance extractor.
            # Target vector y: for each input token, it contains 1 if that token is in the span, 0 otherwise.
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

            # Compute token importances
            out1 = token_importances_extractor.forward(inputs.input_ids,inputs.attention_mask)

            # Calculate factor for teacher forcing
            forcing = 0.5 + np.cos(np.pi*(epoch*len(train_dataloader)+batch_idx)/tot_steps)/2
            importances = forcing * y + (1-forcing) * out1

            # Predict answer
            out2 = encoder_decoder(input_ids = inputs.input_ids,
                         labels = targets.input_ids,
                         token_importances = importances)

            # Obtain loss: sum of the two losses
            loss1 = _loss_function_token_importances_extractor(out1, y)  # Loss on the token importances
            loss2 = out2.loss  # Loss on the answer generation
            loss = loss1 + loss2  # Overall loss

            loss.backward()

            # Update the parameters
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
                    # Compute both the token importances validation loss and the answer generation validation loss
                    val_l1, val_l2 = _loss_validate(tokenizer, token_importances_extractor, encoder_decoder, val_dataloader, disp,
                                                    use_history=use_history, device=device)
                    # Update validation loss history
                    val_loss_history.append([len(loss_history), val_l1.detach().cpu().numpy(), val_l2.detach().cpu().numpy()])
                    torch.cuda.empty_cache()

            # Update training history and print                    
            running_loss1 += loss1.item()
            running_loss2 += loss2.item()
            loss_history.append([loss1.detach().cpu().numpy(), loss2.detach().cpu().numpy()])
            epoch_time = time.time() - start_time
            batch_time = epoch_time/(batch_idx+1)
            disp.update(f"epoch: {epoch + 1}/{epochs}, {batch_idx + 1}/{len(train_dataloader)}, {epoch_time:.0f}s {batch_time*1e3:.0f}ms/step, lr: {optimizer.param_groups[0]['lr']:.3g}, forcing={forcing:.3g}, loss: {running_loss1/batch_steps:.3g} {running_loss2/batch_steps:.3g}")

            # Save checkpoint
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



def _loss_function_token_importances_extractor(probabilities: Tensor, target: Tensor) -> Tensor:
    """Get the loss score of the predictions of Token Importances Extractor module.

    This loss measures the difference between the true span in the passage and the importances over the passage assigned by 
    the extractor.
    In particular, this loss consists in computing the binary cross-entropy, where: the true label of each passage token
    is 1 if that token is in the span, 0 otherwise; the predicted probability is the one assigned by the extractor.
    More precisely, a variant of the binary crossentropy for imbalance classes is used, since there are much more 0 than
    1.

    Going more in depth, the computation is the following.
    - For each token, its binary croessentropy is computed, between its true label and the predicted probability.
    - For each batch sample, the sum of the binary crossentropies of all the tokens is performed.
      Actually, since it is an imbalance problem, a weigthed sum is performed: we normalize the part of the sum related to the
      class 1 by the support of this class, and we do the same for the other class.
    - Finally, we compute the mean across all the batch samples.

    Parameters
    ----------
    probabilities : Tensor
        The token importances predicted probabilities.
    target : Tensor
        The actual span of the passage. For each token, the value is 1 if that token is in the span, 0 otherwise.

    Returns
    -------
    Tensor
        The loss score of the predictions of Token Importances Extractor module.
    """
    loss = - torch.log(probabilities) * (target) / (torch.sum(target, 1, keepdim=True) + 1)
    loss += - torch.log(1 - probabilities) * (1 - target) / (torch.sum(1 - target, 1, keepdim=True) + 1)
    return torch.mean(torch.sum(loss, (1, 2)))



def _loss_validate(tokenizer: PreTrainedTokenizer, token_importances_extractor: TokenImportancesExtractor, 
                   encoder_decoder: EncoderDecoderModel, val_dataloader: DataLoader, disp: DisplayHandle,
                   use_history: bool = False, device: str = 'cuda'):
    """Compute the validation loss of the given model on the given dataset.

    The model is given as its three parts: the tokenizer, the token importances extractor and the encoder-decoder. 

    As for the training, two losses are computed: the one for the token importances and the one for the answer generation.
    So, two validation losses are returned.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizer
        The tokenizer module of the model
    token_importances_extractor : TokenImportancesExtractor
        The tokens importances extractor module of the model
    encoder_decoder : EncoderDecoderModel
        The encoder-decoder module of the model
    val_dataloader : DataLoader
        The validation Dataloader
    disp : DisplayHandle
        The display into which print the results
    use_history : bool, optional
        Whether to use the history or not.
        If True, the history is given as additional input to the model.
    device : str, optional
        Device onto which attach the training, by default 'cuda'

    Returns
    -------
    val_l1: float
        Validation loss1, i.e. the token importances loss.
    val_l2: float
        Validation loss2, i.e. the answer generation loss.
    """

    # Variables for accumulating the loss1 and loss2
    tot_l1=0
    tot_l2=0

    # Number of evaluated samples
    n=0

    # Starting time
    t0=time.time()

    torch.cuda.empty_cache()

    # Iterate over all batches
    for batch_idx, data in enumerate(val_dataloader, 0):
        with torch.no_grad():
            # Get the data
            (passage, question, history), (answer, sep_starts, sep_ends) = data
            history = tuple([h.split(' <sep> ') for h in history])

            # Inject the history into the input
            if use_history:
                separator = f' {tokenizer.sep_token} '
                question = tuple([q + f'{separator if len(h) else ""}' + separator.join(h) for q, h in zip(question, history)])

            # Tokenize the inputs and targets
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

            # Token importances output
            out1 = token_importances_extractor.forward( inputs.input_ids,
                                                        inputs.attention_mask)

            # Compute targets for token importance extractor.
            # Target vector y: for each input token, it contains 1 if that token is in the span, 0 otherwise.
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

            # Compute loss1 (token importances loss)      
            loss1 = _loss_function_token_importances_extractor(out1,y)

            # Compute predicted answer
            out2 = encoder_decoder( input_ids = inputs.input_ids,
                                    attention_mask = inputs.attention_mask,
                                    labels = labels.input_ids,
                                    token_importances = out1)

            # Compute loss2
            loss2 = out2.loss

            # Number of batch samples
            l = (len(question) if isinstance(question,tuple) else 1)

            # Accumulate loss1 and loss2
            tot_l1 += loss1*l
            tot_l2 += loss2*l

            # Update number of evaluated samples
            n += l

        # Print results
        disp.update(f"validate {batch_idx + 1}/{len(val_dataloader)}, {(time.time()-t0):.0f}s {(time.time()-t0) / (batch_idx+1)*1e3:.0f}ms/step, mean losses: {tot_l1/n:.3g} {tot_l2/n:.3g}")

    val_l1, val_l2 = tot_l1 / n, tot_l2 / n

    return val_l1, val_l2
