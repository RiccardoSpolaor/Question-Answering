import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy


def plot_converstion_length_distribution(grouped_df: DataFrameGroupBy) -> None:
    """Analyse the length of the conversation in the given dataset by plotting their distribution.

    Parameters
    ----------
    grouped_df : DataFrameGroupBy
        The dataframe grouped by id, hence by the different conversations
    """
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(15, 7))

    # Show just the x grid
    axes.grid(axis='x')

    # Plot the number of turns in each conversation
    grouped_df['turn_id'].count().hist(bins=25, ax= axes)

    # set title and axis labels
    plt.suptitle('Conversation turns count distribution', x=.5, y=.95, ha='center', fontsize='x-large')
    fig.text(.5, .05, 'number of turns', ha='center')
    fig.text(.075, .5, 'passages count', va='center', rotation='vertical')
    plt.show()

def plot_passage_length_analysis(passages: pd.Series) -> None:
    """Analyse the length of the passages in the given dataset

    Parameters
    ----------
    texts : Series
        The passages.
    """
    # Length of each training sentence
    train_passages_lenghts = list(passages.transform(lambda x: x[0].split(' ')).str.len())

    # Histogram of the sentences length distribution
    hist, bin_edges = np.histogram(train_passages_lenghts, bins=np.max(train_passages_lenghts) + 1, density=True) 
    # Cumulative distribution of the sentences length
    C = np.cumsum(hist)*(bin_edges[1] - bin_edges[0])

    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.plot(bin_edges[1:], hist)
    plt.title('Distribution of the passages length across the train dataset')
    plt.xlabel('Passage length')
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(bin_edges[1:], C)
    plt.title('Cumulative distribution of the passages length across the train dataset')
    plt.xlabel('Passage length')
    plt.grid()
    plt.show()

def plot_answer_span_text_percentile(df: pd.DataFrame) -> None:
    """Analyse the percentile of the answer span text start index with respect to the passage

    Parameters
    ----------
    df : DataFrame
        The dataframe.
    """
    percentiles = df['answer_span_start'] / df['story'].str.len()

    # Histogram of the sentences length distribution
    hist, bin_edges = np.histogram(percentiles.to_list(), bins=50, density=True)
    hist = hist / sum(hist)

    plt.figure(figsize=(15, 7))
    plt.plot(bin_edges[1:], hist)
    plt.title('Distribution of the  percentiles of the answer span text start index with respect to the passage')
    plt.xlabel('Percentile')
    plt.grid()
    plt.show()
