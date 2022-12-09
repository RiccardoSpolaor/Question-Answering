"""Functions taken from [the official evaluation script]
(https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/)
for SQuAD version 2.0.

Modifications
-------------
* Modifications are applied to the name of some functions by adding an underscore to signal that their use is internal.
* Types and docstrings are added to the functions for clear readability.
* Parameter names of the functions are changed for better clarity on their meaning.
* The function `compute_f1` is changed to `_compute_squad_f1`.
* The function `normalize_answer` is rewritten.
* The function `compute_squad_f1` is added.
"""

import collections
import pandas as pd
import re
import string
from typing import List, Union

from models.model import Model

def _normalize_answer(answer: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace from an answer.

    Parameters
    ----------
    answer : str
        Answer to normalize. 
    Returns
    -------
    str
        The normalized answer.
    """
    punctuation_to_exclude = set(string.punctuation)
    articles_regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    
    # Lowercase the string
    answer = answer.lower()
    # Remove punctuation
    answer = ''.join(ch for ch in answer if ch not in punctuation_to_exclude)
    # Remove articles
    answer = re.sub(articles_regex, ' ', answer)
    # Remove extra whitespace
    return ' '.join(answer.split())
    
def _get_tokens(input_string: str) -> List[str]:
    """Get the tokens of a string after it has been normalized.

    Parameters
    ----------
    answer : str
        Answer from which the tokens are obtained. 
    Returns
    -------
    list of strings
        The tokens composing each string.
    """
    if not input_string: 
        return []
    return _normalize_answer(input_string).split()

def _compute_squad_f1(gold_answer: str, predicted_answer: str) -> float:
    """Compute the SQuAD f1 score on a true answer and its prediction.

    Parameters
    ----------
    gold_answer : str
        The true answer.
    predicted_answer : str
        The predicted answer
    Returns
    -------
    float
        The SQuAD f1 score.
    """
    gold_tokens = _get_tokens(gold_answer)
    predicted_tokens = _get_tokens(predicted_answer)
    common = collections.Counter(gold_tokens) & collections.Counter(predicted_tokens)
    num_same = sum(common.values())
    
    if len(gold_tokens) == 0 or len(predicted_tokens) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_tokens == predicted_tokens)
    if num_same == 0:
        return 0
    
    precision = 1.0 * num_same / len(predicted_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def validate(model: Model, validation_dataframe: pd.DataFrame, use_history: bool = False):
    return validation_dataframe.apply(lambda row: 
        compute_squad_f1(row['answer'], 
                         model.generate(row['story'], row['question'], row['history'] if use_history else None)),
        axis=1).mean()

def compute_squad_f1(gold_answers: Union[List[str], str], predicted_answers: Union[List[str], str]) -> float:
    """Compute the average SQuAD f1 score on a series of true and predicted answers.

    Parameters
    ----------
    gold_answers : iterable of str | str
        The true answer.
    predicted_answers : iterable of str | str
        The predicted answers
    Returns
    -------
    float
        The average SQuAD f1 score on the batch.
    """
    # Assert that `gold_answers` and `predicted_answers` are either both iterables or strings.
    assert (isinstance(gold_answers, list) and isinstance(predicted_answers, list)) \
        or (type(gold_answers) == str and type(predicted_answers) == str), \
        '`gold_answers` and `predicted_answers` must be either both iterables or strings'

    # Compute SQuAD f1 score if a single instance is provided.
    if type(gold_answers) == str:
        return _compute_squad_f1(gold_answers, predicted_answers)

    # Compute average SQuAD f1 score if a batch of instances is provided.
    score = 0.
    for g, p in zip(gold_answers, predicted_answers):
        score += _compute_squad_f1(g, p)
    return score / len(gold_answers)
