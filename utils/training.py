import torch 
import time

import matplotlib.pyplot as plt



def train(train_dataloader, model, epochs=(2,1,0), steps_per_update=2, steps_empty_cache=None, device : str ='cpu', plot=False):
    token_importances_extractor = model.token_importances_extractor
    encoder_decoder = model.encoder_decoder
    tokenizer = model.tokenizer

    epochs1 = epochs[0]
    epochs2 = epochs[1]
    epochs3 = epochs[2]

    loss_history1, optim1 = train_tokenImportancesExtractor(train_dataloader, token_importances_extractor, tokenizer, 
                                                  epochs=epochs1, 
                                                  learning_rate=1e-5,
                                                  steps_per_update=steps_per_update, steps_empty_cache=steps_empty_cache,
                                                  device=device)
    if plot and epochs1>0:
        plt.plot(loss_history1)
        plt.xlabel('Epochs')
        plt.title('Training history first phase')

    loss_history2, optim2 = train_EncoderDecoder(train_dataloader, token_importances_extractor, encoder_decoder, tokenizer, 
                                                 epochs=epochs2, 
                         learning_rate=1e-5, steps_per_update=steps_per_update,
                         steps_empty_cache=steps_empty_cache, device=device)
    if plot and epochs2>0:
        plt.plot(loss_history2)
        plt.xlabel('Epochs')
        plt.title('Training history second phase')

    loss_history3, optim3 = train_EncoderDecoder(train_dataloader, token_importances_extractor, encoder_decoder, tokenizer, epochs=epochs3, 
                                       learning_rate=1e-5, 
                                       train_tokenImportancesExtractor=True, 
                                       steps_per_update=steps_per_update, steps_empty_cache=steps_empty_cache, device=device)
    if plot and epochs3>0:
        plt.plot(loss_history3)
        plt.xlabel('Epochs')
        plt.title('Training history third phase')



def loss_func_tokenImportancesExtractor(probs, target):
    loss=  - torch.log(  probs) * (  target) / (torch.sum(  target, 1, keepdim=True)+1)
    loss+= - torch.log(1-probs) * (1-target) / (torch.sum(1-target, 1, keepdim=True)+1)
    return torch.mean(torch.sum(loss,(1,2)))

def train_tokenImportancesExtractor(train_dataloader, token_importances_extractor, tokenizer, epochs=1, learning_rate=1e-5, 
                                    optimizer=None, loss_history=[], steps_per_update=1, steps_empty_cache=None, 
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

            print(f"epoch: {epoch + 1}/{epochs}, {batch_idx + 1}/{len(train_dataloader)}, {epoch_time:.0f}s {batch_time*1e3:.0f}ms/step, lr: {optimizer.param_groups[0]['lr']:.3g}, loss: {running_loss/(batch_idx+1):.3g}               ")#, end = '\r'

        print(f"epoch: {epoch + 1}/{epochs}, {batch_idx + 1}/{len(train_dataloader)}, {epoch_time:.0f}s {batch_time*1e3:.0f}ms/step, lr: {optimizer.param_groups[0]['lr']:.3g}, loss: {running_loss/(batch_idx+1):.3g}              ")

    return loss_history, optimizer


def train_EncoderDecoder(train_dataloader, token_importances_extractor, encoder_decoder, tokenizer, epochs=3, 
                         learning_rate=1e-5, optimizer=None, loss_history=[], train_tokenImportancesExtractor=False, 
                         steps_per_update=1, steps_empty_cache=None, device : str = 'cpu'):

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
            
            print(f"epoch: {epoch + 1}/{epochs}, {batch_idx + 1}/{len(train_dataloader)}, {epoch_time:.0f}s {batch_time*1e3:.0f}ms/step, lr: {optimizer.param_groups[0]['lr']:.3g}, loss: {running_loss/(batch_idx+1):.3g}")

        print(f"epoch: {epoch + 1}/{epochs}, {batch_idx + 1}/{len(train_dataloader)}, {epoch_time:.0f}s {batch_time*1e3:.0f}ms/step, lr: {optimizer.param_groups[0]['lr']:.3g}, loss: {running_loss/(batch_idx+1):.3g}")

    return loss_history, optimizer