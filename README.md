# Question Answering

Question answering through pretrained transformer-based models from Hugging Face.

Question Answering (QA) is the task consisting in generating answers for questions from the passages containing the needed information. Optionally, also the history of previous question.answer turns can be used for producing the answer.

The CoQA dataset has been used: https://stanfordnlp.github.io/coqa/.

In our work, we use a model which consists of two modules: the **tokens importances extractor** and the **encoder-decoder** (i.e. seq2seq). The first module computes an importance score in $[0,1]$ for each passage token, representing the likelihood that the token is in the span of the passage containing the answer. Then, the encoder-decoder takes as additional input these tokens importances, and it generates the answer. The reason of following this approach is to help the encoder-decoder in finding the interesting information in the passage, since it can be very long. Both modules are built from a pre-trained transformer-based architecture, taken from Hugging Face.

Two different pre-trained models have been considered, namely **DistilRoBERTa** and **BERTTiny**. Different random seeds have been set for generating our experiments. Finally, also whether to use or not the conversation history has been taken into account.

For evaluating these different experiments, the average SQuAD F1 score has been computed, both on the validation and test datasets.

## Dependencies
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [Matplotlib](https://pypi.org/project/matplotlib/)
- [NumPy](https://pypi.org/project/numpy/)
- [Pandas](https://pypi.org/project/pandas/)
- [PyTorch](https://pypi.org/project/torch/)

## Repository structure

    .
    ├── coqa    # It contains the dataset files     
    ├── images    # It contains some explanatory images                    
    ├── models     # It contains the models                           
    ├── utils    # It contains the python files with useful functions
    ├── weigths       # It contains the models weigths
    ├── Assignment.ipynb     # Task description
    ├── question answering.ipynb   # Task resolution
    ├── .gitignore
    ├── LICENSE
    ├── report.pdf     # Report of the assignment
    └── README.md

## Versioning

Git is used for versioning.

## Group members

|  Name           |  Surname  |     Email                           |    Username                                             |
| :-------------: | :-------: | :---------------------------------: | :-----------------------------------------------------: |
| Samuele         | Bortolato  | `samuele.bortolato@studio.unibo.it` | [_Sam_](https://github.com/samuele-bortolato)               |
| Antonio         | Politano  | `antonio.politano2@studio.unibo.it` | [_S1082351_](https://github.com/S1082351)               |
| Enrico          | Pittini   | `enrico.pittini@studio.unibo.it`    | [_EnricoPittini_](https://github.com/EnricoPittini)     |
| Riccardo        | Spolaor   | `riccardo.spolaor@studio.unibo.it`  | [_RiccardoSpolaor_](https://github.com/RiccardoSpolaor) |

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
