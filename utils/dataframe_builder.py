import json
import pandas as pd
from typing import List

def _get_history(row: pd.Series) -> List[str]:
    """Get the conversation history as a list of chronological question-answers (e.g.: [question_1, answer_1, question_2, ...]).
    The history corresponds to the questions and answers before the one given in the input pandas `DataFrame` row. Unanswered
    questions along with their respective answer are omitted from the history.

    Parameters
    ----------
    row : Series
        The pandas `DataFrame` row containing the question for which the previous history is built, along with the questions 
        (`row['questions']`) and answers (`row['answers']`) of the passage and the turn id of the question (`row['question_turn_id']`).

    Returns
    -------
    List[str]
        The history of the conversation before a given question.
    """
    history = []
    # Turn id of the current question
    question_turn = row['question_turn_id']

    for q, a in zip(row['questions'], row['answers']):
        # Stop building the history if the turn id of the current question is reached
        if q['turn_id'] >= question_turn:
            break
        # Add the question-answer pair in the history just if the answer is not unknown
        if a['input_text'] != 'unknown':
            history += [q['input_text'], a['input_text']]

    return history


def get_dataframe(json_path: str) -> pd.DataFrame:
    """Build and get the pandas `DataFrame` from a given json path file.

    Parameters
    ----------
    json_path : str
        The path of the json file from which the `DataFrame` is built.

    Returns
    -------
    DataFrame
        The dataframe built from the json file in which each row contains information about a question and the respective 
        answer of a specific conversation:
        * `source`: The source of the passage
        * `id`: The id of the conversation	
        * `filename`: The file name of the passage
        * `story`: The passage considered in the converstion
        * `name`: The name of the file
        * `question_input_text`: The transcription of the question itself
        * `question_turn_id`: The turn of the question itself in the whole conversation
        * `question_bad_turn`: Whether the turn of the question is bad or not
        * `answer_span_start`: The character index in the passage where the answer context starts
        * `answer_span_end`: The character index in the passage where the answer context ends
        * `answer_span_text`: The context of the answer in the passage
        * `answer_input_text`: The answer itself
        * `answer_turn_id`: The turn of the answer itself in the whole conversation (it corresponds to the respective 
          `question_turn_id`)
        * `answer_bad_turn`:  Whether the turn of the answer is bad or not
        * `history`: The history of the conversation before the given question.
    """
    with open(json_path, 'r') as j:
        data = json.loads(j.read())['data']

        # Get the questions and answers dataframes
        questions_df = pd.json_normalize(data, record_path='questions', meta=['id'], record_prefix='question_')
        answers_df = pd.json_normalize(data, record_path='answers', record_prefix='answer_')

        # Get the passages dataframe
        passages_df = pd.json_normalize(data, max_level=0)
        
        # Remove additional answers from the dataframe if present
        if 'additional_answers' in passages_df:
            passages_df.drop('additional_answers', axis=1, inplace=True)

        # Concatenate column-wise the questions and answers dataframes and merge them with the passages dataframe on key `id`
        questions_and_answers_df = pd.concat([questions_df, answers_df], axis=1, join='inner')
        df = passages_df.merge(questions_and_answers_df, on='id')

        # Get the history for each question in  each row of the dataframe
        df['history'] = df.apply(_get_history, axis=1)

        # Remove the questions and answers columns, since they are collected in the history and reset the indices
        df.drop(['questions', 'answers'], axis=1, inplace=True)
        df.reset_index(drop=True)

        return df
