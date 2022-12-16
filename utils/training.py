import os
import pandas as pd
import numpy as np
import torch
import time

import matplotlib.pyplot as plt

from .squad import validate


def train(train_dataloader, val_dataloader, model, model_name, use_history=False, epochs=(2,1,0), optimizers=None, learnin_rates=None, steps_per_update=2, 
          steps_empty_cache=None, seed=None, device : str ='cpu', plot=False):

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

    print('Training phase 1')
    loss_history1 = train_tokenImportancesExtractor(train_dataloader, token_importances_extractor, tokenizer, model_name,
                                                    use_history=use_history, epochs=epochs1, optimizer=optim1, learning_rate=lr1, 
                                                    steps_per_update=steps_per_update, 
                                                    steps_empty_cache=steps_empty_cache, 
                                                    seed=seed, device=device)
    print()
    if plot and epochs1>0:
        plt.plot(loss_history1)
        plt.xlabel('Epochs')
        plt.title('Training history first phase')
    
    # this is completely useless
    #with torch.no_grad():
    #    f1_squad = validate(model, val_dataloader, use_history=use_history)

    #print()
    #print(f'Validation f1 squad after the first phase: {f1_squad}' )
    #print()

    # torch.cuda.empty_cache()

    print('Training phase 2')
    loss_history2 = train_EncoderDecoder(train_dataloader, token_importances_extractor, encoder_decoder, tokenizer, model_name,
                                         use_history=use_history, epochs=epochs2, optimizer=optim2, learning_rate=lr2, 
                                         steps_per_update=steps_per_update, steps_empty_cache=steps_empty_cache, seed=seed,
                                         device=device)
    if plot and epochs2>0:
        plt.plot(loss_history2)
        plt.xlabel('Epochs')
        plt.title('Training history second phase')
    
    with torch.no_grad():
        f1_squad = validate(model, val_dataloader, use_history=use_history)
    
    print()
    print(f'Validation f1 squad after the second phase: {f1_squad}' )
    print()
    # torch.cuda.empty_cache()

    print('Training phase 3')
    loss_history3 = train_EncoderDecoder(train_dataloader, token_importances_extractor, encoder_decoder, tokenizer, model_name,
                                         use_history=use_history, train_tokenImportancesExtractor=True, epochs=epochs3, optimizer=optim3, 
                                         learning_rate=lr3, steps_per_update=steps_per_update, seed=seed,
                                         steps_empty_cache=steps_empty_cache, device=device)
    if plot and epochs3>0:
        plt.plot(loss_history3)
        plt.xlabel('Epochs')
        plt.title('Training history third phase')
    
    if epochs3>0:
        with torch.no_grad():
            f1_squad = validate(model, val_dataloader, use_history=use_history)
        print()
        print(f'Validation f1 squad after the phase 3: {f1_squad}' )
        print()
        
    else:
        print()

    # torch.cuda.empty_cache()



def _save_model_parameters(model, model_name, model_type, seed=None, use_history=False):
    
    if use_history:
        folder_name = './weigths/PQH'
    else:
        folder_name = './weigths/PQ'

    if seed is not None:
        folder_name = os.path.join(folder_name, f'seed{seed}')
    os.makedirs(folder_name, exist_ok=True)
    file_name = f'{model_name}_{model_type}.pt'
    file_path = os.path.join(folder_name, file_name)
    torch.save(model.state_dict(), file_path)

def loss_func_tokenImportancesExtractor(probs, target):
    loss=  - torch.log(  probs) * (  target) / (torch.sum(  target, 1, keepdim=True)+1)
    loss+= - torch.log(1-probs) * (1-target) / (torch.sum(1-target, 1, keepdim=True)+1)
    return torch.mean(torch.sum(loss,(1,2)))

def train_tokenImportancesExtractor(train_dataloader, token_importances_extractor, tokenizer, model_name, use_history=False,
                                    epochs=1, optimizer=None,
                                    learning_rate=1e-5, loss_history=[], steps_per_update=1, steps_empty_cache=None, 
                                    seed=None,
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

            # TODO end
            print(f"epoch: {epoch + 1}/{epochs}, {batch_idx + 1}/{len(train_dataloader)}, {epoch_time:.0f}s {batch_time*1e3:.0f}ms/step, lr: {optimizer.param_groups[0]['lr']:.3g}, loss: {running_loss/(batch_idx+1):.3g}")#, end = '\r')

        print(f"epoch: {epoch + 1}/{epochs}, {batch_idx + 1}/{len(train_dataloader)}, {epoch_time:.0f}s {batch_time*1e3:.0f}ms/step, lr: {optimizer.param_groups[0]['lr']:.3g}, loss: {running_loss/(batch_idx+1):.3g}")

        _save_model_parameters(model=token_importances_extractor, model_name=model_name, model_type='TokenImportancesExtractor',
                                   seed=seed, use_history=use_history)
    return loss_history


def train_EncoderDecoder(train_dataloader, token_importances_extractor, encoder_decoder, tokenizer, model_name, 
                         use_history=False, train_tokenImportancesExtractor=False, epochs=3, optimizer=None, 
                         learning_rate=1e-5, loss_history=[], steps_per_update=1, steps_empty_cache=None, seed=None, 
                         device : str = 'cpu'):

    #token_importances_extractor.to('cuda')
    #encoder_decoder.to('cuda')

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
            
            # TODO end
            print(f"epoch: {epoch + 1}/{epochs}, {batch_idx + 1}/{len(train_dataloader)}, {epoch_time:.0f}s {batch_time*1e3:.0f}ms/step, lr: {optimizer.param_groups[0]['lr']:.3g}, loss: {running_loss/(batch_idx+1):.3g}")#, end='\r')

        print(f"epoch: {epoch + 1}/{epochs}, {batch_idx + 1}/{len(train_dataloader)}, {epoch_time:.0f}s {batch_time*1e3:.0f}ms/step, lr: {optimizer.param_groups[0]['lr']:.3g}, loss: {running_loss/(batch_idx+1):.3g}")

        _save_model_parameters(model=encoder_decoder, model_name=model_name, model_type='EncoderDecoder', 
                                   seed=seed, use_history=use_history)
        if train_tokenImportancesExtractor:
            _save_model_parameters(model=token_importances_extractor, model_name=model_name, model_type='TokenImportancesExtractor', 
                                    seed=seed, use_history=use_history)

    return loss_history


def train_(train_dataloader, val_dataloader, model, model_name, use_history=False,
            epochs=3, learning_rate=5e-5, optimizer=None, forcing_long_tail=1.5,
            steps_per_update=1, steps_empty_cache=None, steps_validate=None,
            loss_history=None, val_loss_history=None,
            seed=None, device ='cpu'):

    train_extractor=True
    max_p = 1
    p=max_p
    

    if use_history:
        folder_name = 'weigths\PQH'
    else:
        folder_name = 'weigths\PQ'

    if seed is not None:
        folder_name = os.path.join(folder_name, f'seed{seed}')
    os.makedirs(folder_name, exist_ok=True)
    file_name = f'{model_name}_temp_extractor.pt'
    file_path = os.path.join(folder_name, file_name)
    
    if loss_history is None:
        loss_history=[]
        val_loss_history=[]

    token_importances_extractor = model.token_importances_extractor
    encoder_decoder = model.encoder_decoder
    tokenizer = model.tokenizer


    if optimizer is None:
        optimizer = torch.optim.Adam(iter(list(token_importances_extractor.parameters())+list(encoder_decoder.parameters())), 
                                        lr=learning_rate)

    tot_steps=len(train_dataloader)*epochs

    for epoch in range(epochs):  # loop over the dataset multiple times
        min_val_l1 = np.inf
        torch.cuda.empty_cache()
        running_loss1 = 0.0
        running_loss2 = 0.0
        optimizer.zero_grad()
        start_time = time.time()
        for batch_idx, data in enumerate(train_dataloader, 0):
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

            loss1 = loss_func_tokenImportancesExtractor(out1,y)

            forcing = 0.5 - np.cos( np.pi*np.power( 1 - (len(train_dataloader)*epoch+batch_idx)/tot_steps, forcing_long_tail ) )/2
            importances = forcing*y + (1-forcing)*out1

            out2 = encoder_decoder( input_ids = inputs.input_ids,
                                    attention_mask = inputs.attention_mask,
                                    labels = labels.input_ids,
                                    token_importances = importances.detach())

            loss2 = out2.loss

            if train_extractor:
                loss = loss1 + loss2
            else:
                loss = loss2

            loss.backward()

            if batch_idx % steps_per_update == steps_per_update-1:
                optimizer.step()
                optimizer.zero_grad()

            if steps_empty_cache is not None:
                if batch_idx % steps_empty_cache == steps_empty_cache-1:
                    torch.cuda.empty_cache()

            if steps_validate is not None:
                if batch_idx % steps_validate == steps_validate-1:
                    torch.cuda.empty_cache()

                    val_l1, val_l2 = loss_validate( tokenizer, token_importances_extractor, encoder_decoder, val_dataloader,
                                                    use_history=use_history, device=device)

                    val_loss_history.append([len(loss_history), val_l1.detach().cpu().numpy(), val_l2.detach().cpu().numpy()])
                    
                    
                    if val_l1<min_val_l1:
                        min_val_l1=val_l1
                        torch.save(token_importances_extractor.state_dict(), file_path)
                        p=max_p
                    else:
                        token_importances_extractor.load_state_dict(torch.load(file_path))
                        p-=1
                        print()
                        train_extractor=False
                        if p<0:
                            train_extractor=False
                    torch.cuda.empty_cache()
                    
            running_loss1 += loss1.item()
            running_loss2 += loss2.item()

            loss_history.append([loss1.detach().cpu().numpy(), loss2.detach().cpu().numpy()])

            epoch_time = time.time() - start_time
            batch_time = epoch_time/(batch_idx+1)
            
            # TODO end
            print(f"epoch: {epoch + 1}/{epochs}, {batch_idx + 1}/{len(train_dataloader)}, {epoch_time:.0f}s {batch_time*1e3:.0f}ms/step, lr: {optimizer.param_groups[0]['lr']:.3g}, forcing={forcing:.3g}, losses: {running_loss1/(batch_idx+1):.3g}  {running_loss2/(batch_idx+1):.3g}".ljust(120), end='\r')#)

        print(f"epoch: {epoch + 1}/{epochs}, {batch_idx + 1}/{len(train_dataloader)}, {epoch_time:.0f}s {batch_time*1e3:.0f}ms/step, lr: {optimizer.param_groups[0]['lr']:.3g}, forcing={forcing:.3g}, losses:{running_loss1/(batch_idx+1):.3g}  {running_loss2/(batch_idx+1):.3g}".ljust(120))

    torch.cuda.empty_cache()

    val_l1, val_l2 = loss_validate( tokenizer, token_importances_extractor, encoder_decoder, val_dataloader,
                                    use_history=use_history, device=device)
    val_loss_history.append([len(loss_history), val_l1.detach().cpu().numpy(), val_l2.detach().cpu().numpy()])

    if val_l1<min_val_l1:
        min_val_l1=val_l1
        torch.save(token_importances_extractor.state_dict(), file_path)
        p=max_p
    else:
        print()
        token_importances_extractor.load_state_dict(torch.load(file_path))
        p-=1
        train_extractor=False
        if p<0:
            train_extractor=False
    torch.cuda.empty_cache()

    print(f'best validation loss1: {min_val_l1}')

    token_importances_extractor.load_state_dict(torch.load(file_path))

    os.remove(file_path)

    file_name = f'{model_name}.pt'
    file_path = os.path.join(folder_name, file_name)
    torch.save(model.state_dict(), file_path)

    return loss_history, val_loss_history, optimizer




from utils.squad import _compute_squad_f1

def loss_validate(  tokenizer, token_importances_extractor, encoder_decoder, val_dataloader,
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

            loss1 = loss_func_tokenImportancesExtractor(out1,y)

            out2 = encoder_decoder( input_ids = inputs.input_ids,
                                    attention_mask = inputs.attention_mask,
                                    labels = labels.input_ids,
                                    token_importances = out1)
            
            loss2 = out2.loss

            l=(len(question) if isinstance(question,tuple) else 1)
            tot_l1 += loss1*l
            tot_l2 += loss2*l
            n += l

        print(f"validate {batch_idx + 1}/{len(val_dataloader)}, {(time.time()-t0):.0f}s {(time.time()-t0)/(batch_idx+1)*1e3:.0f}ms/step, mean losses: {tot_l1/n:.3g} {tot_l2/n:.3g}".ljust(120), end='\r')
    
    return tot_l1/n, tot_l2/n