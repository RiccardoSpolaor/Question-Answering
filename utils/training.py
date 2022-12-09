import os
import torch 
import time

import matplotlib.pyplot as plt



def train(train_dataloader, model, model_name, epochs=(2,1,0), optimizers=None, learnin_rates=None, steps_per_update=2, 
          steps_empty_cache=None, parameters_file_end='', device : str ='cpu', plot=False):

    token_importances_extractor = model.token_importances_extractor
    encoder_decoder = model.encoder_decoder
    tokenizer = model.tokenizer

    epochs1 = epochs[0]
    epochs2 = epochs[1]
    epochs3 = epochs[2]

    if optimizers is not None:
        optim1 = optimizers[0]
        optim2 = optimizers[1]
        optim3 = optimizers[2]
    else:
        optim1, optim2, optim3 = None, None, None

    if learnin_rates is not None:
        lr1 = learnin_rates[0]
        lr2 = learnin_rates[1]
        lr3 = learnin_rates[2]
    else:
        lr1, lr2, lr3 = 1e-5, 1e-5, 1e-5

    loss_history1 = train_tokenImportancesExtractor(train_dataloader, token_importances_extractor, tokenizer, model_name,
                                                    epochs=epochs1, optimizer=optim1, learning_rate=lr1, 
                                                    steps_per_update=steps_per_update, 
                                                    steps_empty_cache=steps_empty_cache, parameters_file_end=parameters_file_end,
                                                    device=device)
    if plot and epochs1>0:
        plt.plot(loss_history1)
        plt.xlabel('Epochs')
        plt.title('Training history first phase')

    # torch.cuda.empty_cache()

    loss_history2 = train_EncoderDecoder(train_dataloader, token_importances_extractor, encoder_decoder, tokenizer, model_name,
                                         epochs=epochs2, optimizer=optim2, learning_rate=lr2, 
                                         steps_per_update=steps_per_update, steps_empty_cache=steps_empty_cache, 
                                         parameters_file_end=parameters_file_end,
                                         device=device)
    if plot and epochs2>0:
        plt.plot(loss_history2)
        plt.xlabel('Epochs')
        plt.title('Training history second phase')

    # torch.cuda.empty_cache()

    loss_history3 = train_EncoderDecoder(train_dataloader, token_importances_extractor, encoder_decoder, tokenizer, model_name,
                                         train_tokenImportancesExtractor=True, epochs=epochs3, optimizer=optim3, 
                                         learning_rate=lr3, steps_per_update=steps_per_update, 
                                         steps_empty_cache=steps_empty_cache, device=device)
    if plot and epochs3>0:
        plt.plot(loss_history3)
        plt.xlabel('Epochs')
        plt.title('Training history third phase')

    # torch.cuda.empty_cache()



def _save_model_parameters(model, model_name, model_type, suffix=''):
    os.makedirs('./weigths', exist_ok=True)
    file_name = f'{model_name}_{model_type}_{suffix}.pt'
    file_path = os.path.join('./weigths', file_name)
    torch.save(model.state_dict(), file_path)

def loss_func_tokenImportancesExtractor(probs, target):
    loss=  - torch.log(  probs) * (  target) / (torch.sum(  target, 1, keepdim=True)+1)
    loss+= - torch.log(1-probs) * (1-target) / (torch.sum(1-target, 1, keepdim=True)+1)
    return torch.mean(torch.sum(loss,(1,2)))

def train_tokenImportancesExtractor(train_dataloader, token_importances_extractor, tokenizer, model_name, epochs=1, 
                                    optimizer=None,
                                    learning_rate=1e-5, loss_history=[], steps_per_update=1, steps_empty_cache=None, 
                                    parameters_file_end='',
                                    device : str = 'cpu'):
    
    #token_importances_extractor.to('cuda')

    if optimizer is None:
        optimizer = torch.optim.AdamW(iter(list(token_importances_extractor.parameters())), lr=learning_rate)
    
    for epoch in range(epochs):  # loop over the dataset multiple times
    
        torch.cuda.empty_cache()
        running_loss = 0.0
        optimizer.zero_grad()
        start_time = time.time()
        for batch_idx, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            (passage, question, history), (_, sep_starts, sep_ends) = data
            history = (h.split(' <sep> ') for h in history)

            inputs = tokenizer(
                        question,
                        passage,
                        max_length=512,
                        truncation=True,
                        padding=True,
                        return_tensors="pt",
                    ).to(device)
            
            pred = token_importances_extractor.forward(inputs.input_ids,
                                inputs.attention_mask)

            y = torch.zeros(inputs.input_ids.shape+(1,), device=pred.device)

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
                

            loss = loss_func_tokenImportancesExtractor(pred,y)

            loss.backward()
            
            if batch_idx % steps_per_update == steps_per_update-1:
                optimizer.step()
                optimizer.zero_grad()

            if steps_empty_cache is not None:
                if batch_idx % steps_empty_cache == steps_empty_cache-1:
                    torch.cuda.empty_cache()

            # print statistics
            running_loss += loss.item()

            loss_history.append(loss.detach().cpu().numpy())
            
            epoch_time = time.time() - start_time
            batch_time = epoch_time/(batch_idx+1)

            print(f"epoch: {epoch + 1}/{epochs}, {batch_idx + 1}/{len(train_dataloader)}, {epoch_time:.0f}s {batch_time*1e3:.0f}ms/step, lr: {optimizer.param_groups[0]['lr']:.3g}, loss: {running_loss/(batch_idx+1):.3g}", end = '\r')

            _save_model_parameters(model=token_importances_extractor, model_name=model_name, model_type='TokenImportancesExtractor',
                                suffix=parameters_file_end)

        print(f"epoch: {epoch + 1}/{epochs}, {batch_idx + 1}/{len(train_dataloader)}, {epoch_time:.0f}s {batch_time*1e3:.0f}ms/step, lr: {optimizer.param_groups[0]['lr']:.3g}, loss: {running_loss/(batch_idx+1):.3g}")

    return loss_history


def train_EncoderDecoder(train_dataloader, token_importances_extractor, encoder_decoder, tokenizer, model_name,
                         train_tokenImportancesExtractor=False, epochs=3, optimizer=None, learning_rate=1e-5, loss_history=[],  
                         steps_per_update=1, steps_empty_cache=None, parameters_file_end='', device : str = 'cpu'):

    #token_importances_extractor.to('cuda')
    #encoder_decoder.to('cuda')

    encoder_decoder.config.decoder_start_token_id = tokenizer.cls_token_id
    encoder_decoder.config.pad_token_id = tokenizer.pad_token_id

    if optimizer is None:
        if train_tokenImportancesExtractor:
            optimizer = torch.optim.Adam(iter(list(token_importances_extractor.parameters())+list(encoder_decoder.parameters())), 
                                         lr=learning_rate)
        else:
            optimizer = torch.optim.Adam(iter(list(encoder_decoder.parameters())), lr=learning_rate)

    for epoch in range(epochs):  # loop over the dataset multiple times
    
        torch.cuda.empty_cache()
        running_loss = 0.0
        optimizer.zero_grad()
        start_time = time.time()
        for batch_idx, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            (passage, question, history), (answer, _, _) = data
            history = (h.split(' <sep> ') for h in history)
            
            inputs = tokenizer(
                        question,
                        passage,
                        max_length=512,
                        truncation=True,
                        padding=True,
                        return_tensors="pt",
                    ).to(device)

            labels = tokenizer(
                answer,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(device)

            if train_tokenImportancesExtractor:
                token_importances_output = token_importances_extractor.forward(inputs.input_ids,
                                inputs.attention_mask)
            else:
                with torch.no_grad():
                    token_importances_output = token_importances_extractor.forward(inputs.input_ids,
                                    inputs.attention_mask)

            encoder_decoder_output = encoder_decoder(input_ids = inputs.input_ids,
                         labels = labels.input_ids,
                         token_importances = token_importances_output)
            
            loss = encoder_decoder_output.loss
            loss.backward()

            if batch_idx % steps_per_update == steps_per_update-1:
                optimizer.step()
                optimizer.zero_grad()

            if steps_empty_cache is not None:
                if batch_idx % steps_empty_cache == steps_empty_cache-1:
                    torch.cuda.empty_cache()

            running_loss += loss.item()

            loss_history.append(loss.detach().cpu().numpy())
            
            epoch_time = time.time() - start_time
            batch_time = epoch_time/(batch_idx+1)
            
            print(f"epoch: {epoch + 1}/{epochs}, {batch_idx + 1}/{len(train_dataloader)}, {epoch_time:.0f}s {batch_time*1e3:.0f}ms/step, lr: {optimizer.param_groups[0]['lr']:.3g}, loss: {running_loss/(batch_idx+1):.3g}", end='\r')

            _save_model_parameters(model=encoder_decoder, model_name=model_name, model_type='EncoderDecoder', 
                                   suffix=parameters_file_end)
            if train_tokenImportancesExtractor:
                _save_model_parameters(model=token_importances_extractor, model_name=model_name, model_type='TokenImportancesExtractor', 
                                       suffix=parameters_file_end)

        print(f"epoch: {epoch + 1}/{epochs}, {batch_idx + 1}/{len(train_dataloader)}, {epoch_time:.0f}s {batch_time*1e3:.0f}ms/step, lr: {optimizer.param_groups[0]['lr']:.3g}, loss: {running_loss/(batch_idx+1):.3g}")

    return loss_history