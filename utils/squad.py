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
import numpy as np
import re
import string
import time
import torch
from torch.utils.data import DataLoader
from typing import List

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

def compute_squad_f1(gold_answer: str, predicted_answer: str) -> float:
    """Compute the SQuAD f1 score on a true answer and its prediction.

    Parameters
    ----------
    gold_answer : str
        The true answer.
    predicted_answer : str
        The predicted answer.
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
        return 0.
    
    precision = 1.0 * num_same / len(predicted_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def validate(model: Model, val_dataloader: DataLoader, use_history: bool) -> float:
    """Evaluate the model on the validation dataset according to the average SQuAD F1 score.

    Parameters
    ----------
    model : Model
        The model to evaluate.
    val_dataloader : DataLoader
        The validation dataloader.
    use_history : bool
        Whether to use the previous QaA discussion history or not.

    Returns
    -------
    float
        The average SQuAD F1 score of the validation dataset.
    """
    
    tot_squad_f1 = 0.
    total_n_instances = 0
    starting_time = time.time()

    torch.cuda.empty_cache()

    for batch_idx, data in enumerate(val_dataloader, 0):
        with torch.no_grad():
            (passages, questions, histories), (answers, _, _) = data
            
            predictions = model.generate(passages, questions, histories if use_history else None)
            
            tot_squad_f1 += np.sum([compute_squad_f1(gold, predicted) for gold, predicted in zip(answers, predictions)])
            total_n_instances += len(questions) if isinstance(questions, tuple) else 1

        print(f"{batch_idx + 1}/{len(val_dataloader)}, {(time.time() - starting_time):.0f}s",
              f"{(time.time()-starting_time)/(batch_idx+1)*1e3:.0f}ms/step, mean SQuAD F1: {tot_squad_f1/total_n_instances}",
              end='\r')
    
    return tot_squad_f1 / total_n_instances
