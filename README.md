# argminer

![GitHub](https://img.shields.io/github/license/namiyousef/argument-mining)
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/namiyousef/argument-mining/Python%20package)
![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/namiyousef/argument-mining)
---

_argminer_ is a PyTorch based package for argument mining on state-of-the-art datasets. It provides a high level API for processing these datasets and applying different labelling strategies, augmenting them, training on them using a model from [huggingface](https://huggingface.co/), and performing inference on them.

## Datasets

- Argument Annotated Essays [[1]](#1): a collection of 402 essays written by college students containing with clauses classified as Premise, Claim and MajorClaim. This is the SOTA dataset in the filed of argument mining.

- PERSUADE [[2]](#2): A newly released dataset from the Feedback Prize Kaggle [competition](https://www.kaggle.com/competitions/feedback-prize-2021/overview). This is a collection of over 15000 argumentative essays written by U.S. students in grades 6-12. It contains the following argument types: Lead, Position, Claim, Counterclaim, Rebuttal, Evidence, and Concluding Statement.

Another important dataset in this field is the ARG2020 [[3]](#3) dataset. This was made public at a time coinciding with the early access release so support for it does not yet exist. There are plans to implement this in the future.

## Data Processing API

Datasets in the field of argument mining are stored in completely different formats and have different labels. This makes it difficult to compare and contrast model performance on them provided different configurations and processing steps. 

The data processing API provides a standard method of processing these datasets end-to-end to enable fast and consistent experimentation and modelling.

- ADD DIAGRAM

### Labelling Strategies

SOTA argument mining methods treat the process of chunking text into it's argument types and classifying them as an NER problem on long sequences. This means that segment of text with it's associated label is converted into a sequence of classes for the model to predict. To illustrate this we will be using the following passage:

"Natural Language Processing is the best field in Machine Learning. According to a recent poll by DeepLearning.ai it's popularity has increased by twofold in the last 2 years."

Let's suppose for the sake of argument that the passage has the following labels:
- "Natural Language Processing is the best field in Machine Learning.": Claim
- "According to a recent poll by DeepLearning.ai it's popularity has increased by twofold in the last 2 years.": Evidence

The data processing API allows multiple labelling schemes. These are best


### Data Augmentation (Adversarial Examples)

### Evaluation


# Installation
You can install the package directly from PyPI using pip:
```bash
pip install argminer
```
If looking to install from the latest commit, please use the following:
```bash
pip install argminer@git+https://git@github.com/namiyousef/argument-mining.git@develop 
```

# Available Datasets
- TODO

# Documentation
- TODO

# Quick Start
- TODO



# References
<a id="1">[1]</a>
Christian S. and Iryna G. (2017). _Parsing Argumentation Structures in Persuasive Essays_. DOI: 10.1162/COLI_a_00295

<a id="2">[2]</a> 
Scott C. and the Learning Agency. url: https://github.com/scrosseye/PERSUADE_corpus

<a id="3">[3]</a> 
Alhindi, T. and Ghosh, D. (2021). 
"Sharks are not the threat humans are": _Argument Component Segmentation in School Student Essays_. Proceedings of the 16th Workshop on Innovative Use of NLP for Building Educational Applications. p.210-222. url: https://aclanthology.org/2021.bea-1.22