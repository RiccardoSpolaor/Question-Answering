import matplotlib.pyplot as plt
import numpy as np
import torch


def _plot_loss_subplot(training_loss_history: np.ndarray, validation_loss_history: np.ndarray, n_batches: int,
                       subplot_index: int, plot_token_importances_extractor: bool = True,
                       use_log_scale: bool = False) -> None:
    """Plot the subplot of the loss history for a certain module.

    Parameters
    ----------
    training_loss_history : ndarray
        The training loss history.
    validation_loss_history : np.ndarray
        The validation loss history.
    n_batches : int
        The number of batches used to show the normalized training loss history.
    subplot_index : int
        Current index of the subplot.
    plot_token_importances_extractor : bool, optional
        If True, show the loss history of the Token Importances Extractor module. Otherwise of the Seq2seq module.
        Defaults to True.
    use_log_scale : bool, optional
        Whether to show the plot in a logarithmic scale or not. Defaults to False.
    """
    plt.subplot(2, 2, subplot_index)

    # Get the indices of the train and validation loss of the token importances extractor module 
    # (if `plot_token_importances_extractor` is True) or the Seq2seq module (if the same variable is false). 
    loss_index = 0 if plot_token_importances_extractor else 1
    val_loss_index = 1 if plot_token_importances_extractor else 2

    module_name = 'Token Importances Extractor' if plot_token_importances_extractor else 'Seq2seq'

    plt.title(f'{module_name} Loss History{" using log scale" if use_log_scale else ""}')

    plt.plot(training_loss_history[:,loss_index], label='Training loss')
    plt.plot(np.convolve(training_loss_history[:,loss_index], np.ones(n_batches)/n_batches, mode='valid'), 
             label=f'Training loss averaged on {n_batches} batches')

    # Plot validation history if present
    if len(validation_loss_history) > 0:
        plt.plot(validation_loss_history[:,0], validation_loss_history[:,val_loss_index], 'r*', 
                 label='Validation loss')
    
    plt.xlabel('iterations')
    
    # Use log scale if specified
    if use_log_scale:
        plt.yscale('log')
        plt.ylabel('loss (log)')
    else:
        plt.ylabel('loss')

    plt.legend()

def plot_training_history(checkpoints_path: str) -> None:
    """Plot the training history of a model.

    Parameters
    ----------
    checkpoints_path : str
        The path of the checkpoints of a model for which the training history is plotted.
    """
    # Load model checkpoints
    checkpoint = torch.load(checkpoints_path)

    # Get loss history and validation loss history
    loss_history = checkpoint['loss_history']
    validation_loss_history = checkpoint['val_loss_history']

    n_batches = 100

    plt.figure(figsize=(15,12))
    plt.subplot(2,2,1)
    plt.suptitle('Training procedure analysis')
    
    # Plot loss history of the Token Importances Extractor module
    _plot_loss_subplot(loss_history, validation_loss_history, n_batches, 1, plot_token_importances_extractor=True,
                       use_log_scale=False)

    # Plot loss history of the Seq2seq module
    _plot_loss_subplot(loss_history, validation_loss_history, n_batches, 2, plot_token_importances_extractor=False,
                       use_log_scale=False)

    # Plot loss history of the Token Importances Extractor module in log scale
    _plot_loss_subplot(loss_history, validation_loss_history, n_batches, 3, plot_token_importances_extractor=True,
                       use_log_scale=True)

    # Plot loss history of the Seq2seq module in log scale
    _plot_loss_subplot(loss_history, validation_loss_history, n_batches, 4, plot_token_importances_extractor=False,
                       use_log_scale=True)
    
    plt.show()
