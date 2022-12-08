import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from typing import List, Tuple

class _Dataset(torch.utils.data.Dataset):
    """Class extending the torch `Dataset`."""

    def __init__(self, df: pd.DataFrame) -> None:
        """Create an instance of the dataset

        Parameters
        ----------
        df : DataFrame
            The pandas `DataFrame` from which the dataset is created
        """
        self.df = df.copy()
        self.df.reset_index(drop=True, inplace=True)

    def __len__(self) -> int:
        """Get the length of the dataset

        Returns
        -------
        int
            The length of the dataset.
        """
        return len(self.df)

    def __getitem__(self, index: int) -> Tuple[Tuple[str, str, List[str]], str]:
        """Get an instance from the dataset

        Parameters
        ----------
        index : int
            The index of the item in the dataset

        Returns
        -------
        (str, str, List[str]), str
            A tuple of two elements containing: 
            * A tuple of the passage, the question and the history as its first element
            * The answer as its second element
        """
        # Get the row at index `index`
        row = self.df.iloc[index]

        # Get information
        passage = row['story']
        question = row['question']
        answer = row['answer']
        history = row['history']
        span_start = row['answer_span_start']
        span_end = row['answer_span_end']
        
        return (passage, question, ' <sep> '.join(history)), (answer, span_start, span_end)

def get_dataloader(df: pd.DataFrame, batch_size: int = 16, shuffle: bool = True) -> DataLoader:
    """Get a dataloader for a given dataframe.

    Parameters
    ----------
    df: DataFrame
        The dataframe from which the dataloader is created.
    batch_size : int, optional
        The batch size to consider. Defaults to 16.
    shuffle : bool, optional
        Whether to shuffle the data or not. Defaults to True.

    Returns
    -------
    Dataloader
        The dataloader.
    """
    return DataLoader(_Dataset(df), batch_size=batch_size, shuffle=shuffle)