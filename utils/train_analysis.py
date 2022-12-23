import numpy as np
import os
import torch
import matplotlib.pyplot as plt

def _plot_loss_subplot(training_loss_history: np.ndarray, validation_loss_history: np.ndarray, N: int, subplot_index: int,
                       plot_token_importances_extractor: bool = True, use_log_scale: bool = False):
    plt.subplot(2, 2, subplot_index)

    # Get the indices of the train and validation loss of the token importances extractor module 
    # (if `plot_token_importances_extractor` is True) or the encoder decoder module (if the same variable is false). 
    loss_index = 0 if plot_token_importances_extractor else 1
    val_loss_index = 1 if plot_token_importances_extractor else 2

    plt.plot(training_loss_history[:,loss_index])
    plt.plot(np.convolve(training_loss_history[:,loss_index], np.ones(N)/N, mode='valid'))

    # Plot validation history if present
    if len(validation_loss_history) > 0:
        plt.plot(validation_loss_history[:,0], validation_loss_history[:,val_loss_index], 'r*')
    
    # Use log scale if specified
    if use_log_scale:
        plt.yscale('log')

def plot_training_history(folder_name: str, model_name: str):
    # Load model checkpoints
    checkpoint = torch.load(os.path.join(folder_name, f"{model_name.replace('/','_')}.pth"))

    # Get loss history and validation loss history
    loss_history = checkpoint['loss_history']
    validation_loss_history = checkpoint['val_loss_history']

    N = 100

    plt.figure(figsize=(15,12))
    plt.subplot(2,2,1)
    
    # Plot loss history of the token 
    _plot_loss_subplot(loss_history, validation_loss_history, N, 1, plot_token_importances_extractor=True, use_log_scale=False)
    
    _plot_loss_subplot(loss_history, validation_loss_history, N, 2, plot_token_importances_extractor=False, use_log_scale=False)

    _plot_loss_subplot(loss_history, validation_loss_history, N, 3, plot_token_importances_extractor=True, use_log_scale=True)

    _plot_loss_subplot(loss_history, validation_loss_history, N, 4, plot_token_importances_extractor=False, use_log_scale=True)
    
    
    '''plt.plot(lh[:,0])
    plt.plot(np.convolve(lh[:,0], np.ones(N)/N, mode='valid'))
    # Plot validation history if present
    if len(vlh) > 0:
        plt.plot(vlh[:,0], vlh[:,1], 'r*')

    plt.subplot(2,2,2)
    plt.plot(lh[:,1])
    plt.plot(np.convolve(lh[:,1], np.ones(N)/N, mode='valid'))
    if len(vlh) > 0:
        plt.plot(vlh[:,0], vlh[:,2], 'r*')

    plt.subplot(2,2,3)
    plt.plot(lh[:,0])
    plt.plot(np.convolve(lh[:,0], np.ones(N)/N, mode='valid'))
    if len(vlh) > 0:
        plt.plot(vlh[:,0], vlh[:,1], 'r*')
    plt.yscale('log')

    plt.subplot(2,2,4)
    plt.plot(lh[:,1])
    plt.plot(np.convolve(lh[:,1], np.ones(N)/N, mode='valid'))
    if len(vlh) > 0:
        plt.plot(vlh[:,0], vlh[:,2], 'r*')
    plt.yscale('log')'''
    plt.show()
