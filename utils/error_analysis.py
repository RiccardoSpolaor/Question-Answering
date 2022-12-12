import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from typing import List, Tuple

from models.model import Model

from utils.dataloader_builder import get_dataloader
from utils.squad import _compute_squad_f1


def get_worst_answers(model: Model, df_source: pd.DataFrame, use_history: bool = False, k: int = 5, 
                      min_answer_length: int = 1,
                      evaluation_batch_size: int = 16) -> List[Tuple[float, str, str, str, str, str]]:
    # Element structure: (f1 SQuAD, question, passage, history, gold answer, predicted answer, span_start, span_end)
    worst_answers = []

    torch.cuda.empty_cache()

    source_dataloader = get_dataloader(df=df_source, batch_size=evaluation_batch_size)

    for data in tqdm(source_dataloader):

        with torch.no_grad():
            # get the inputs; data is a list of [inputs, labels]
            (passage, question, history), (answer, span_start, span_end) = data

            pred = model.generate(passage, question, history if use_history else None)

            if min_answer_length > 1:
                mask = np.array([len(predicted.split(' ')) >= min_answer_length for predicted in pred])
                passage = np.array(passage)[mask]
                question = np.array(question)[mask]
                history = np.array(history)[mask]
                answer = np.array(answer)[mask]
                pred = np.array(pred)[mask]

            f1_scores = np.array([_compute_squad_f1(gold, predicted) for gold, predicted in zip(answer,pred)])
            samples_indices = np.argsort(f1_scores)[:k]

            worst_answers += [tuple([f1_scores[sample_idx], question[sample_idx], passage[sample_idx], history[sample_idx], 
                                     answer[sample_idx], pred[sample_idx], span_start[sample_idx], span_end[sample_idx]]) 
                              for sample_idx in samples_indices]
            worst_answers = sorted(worst_answers)[:k]

    return worst_answers

def show_worst_errors(source_name: str, sources_statistics_dict: dict, show_history: bool = False) -> None:
    results = sources_statistics_dict[source_name]
    for r in results:
        f1_squad, question, passage, history, gold_answer, predicted_answer, _, _ = r
        
        print(f'* Passage: "{passage[:min(100, len(passage))]}..."')
        
        print(f'* Question: "{question}"')
        
        if show_history:
            history = history.split('<sep>')
            if len(history) == 0:
                print('* History: N/A')
            else:
                questions = history[::2]
                answers = history[1::2]
                questions_and_answers = [f'Q{i+1}: "{q}"; A{i+1}: "{a}"' for i, (q, a) in enumerate(zip(questions, answers))]
                questions_and_answers = questions_and_answers[-min(2, len(questions_and_answers)):]
                history_string = '; '.join(questions_and_answers)
                print(f'* History: {history_string}')

        print(f'* Gold Answer: "{gold_answer}"')

        print(f'* Predicted Answer: "{predicted_answer}"')

        print(f'* F1 SQuAD: {f1_squad}')
        print()

def plot_token_importances(source_name: str, sources_statistics_dict: dict, model: Model, 
                           use_history: bool = False) -> None:
    results = sources_statistics_dict[source_name]

    n_results = len(results)
    n_cols = 2

    # Compute number of rows in the plot
    n_rows = n_results // n_cols 

    # Add obe row if necessary
    if n_results % n_cols != 0:
        n_rows += 1

    # Create a position index
    position_range = np.arange(1, n_results + 1)

    fig = plt.figure(figsize=(15, 10))
    
    fig.suptitle(f'Token importances of the {n_results} worst results.')

    for i in range(n_results):
        _, question, passage, history, _, _, span_start, span_end = results[i]

        token_importances = model.compute_token_importances(passage, question, span_start, span_end,
                                                            history if use_history else None)

        y = np.zeros(shape=(token_importances.shape[1],))

        y[span_start : span_end] = 1

        ax = fig.add_subplot(n_rows,n_cols, position_range[i])
        ax.plot(token_importances.cpu().detach()[0], label='Predicted token importances')
        ax.plot(y, label='Golden token importances')
        ax.set_xlabel('Token ids')
        ax.set_ylabel('Importances')
        ax.set_title(f'Result {i + 1}')
        ax.legend()

    plt.subplots_adjust(hspace=.5)

    plt.show()
