# ArgMiner: End-to-end Argument Mining 

![GitHub](https://img.shields.io/github/license/namiyousef/argument-mining)
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/namiyousef/argument-mining/Python%20package)
![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/namiyousef/argument-mining)
---

_argminer_ is a PyTorch based package for argument mining on state-of-the-art datasets. It provides a high level API for processing these datasets and applying different labelling strategies, augmenting them, training on them using a model from [huggingface](https://huggingface.co/), and performing inference on them.

## Datasets

- Argument Annotated Essays [[1]](#1): a collection of 402 essays written by college students containing with clauses classified as Premise, Claim and MajorClaim. This is the SOTA dataset in the filed of argument mining.

- PERSUADE [[2]](#2): A newly released dataset from the Feedback Prize Kaggle [competition](https://www.kaggle.com/competitions/feedback-prize-2021/overview). This is a collection of over 15000 argumentative essays written by U.S. students in grades 6-12. It contains the following argument types: Lead, Position, Claim, Counterclaim, Rebuttal, Evidence, and Concluding Statement.

Another important dataset in this field is the ARG2020 [[3]](#3) dataset. This was made public at a time coinciding with the early access release so support for it does not yet exist. There are plans to implement this in the future.

## Installation
You can install the package directly from PyPI using pip:
```bash
pip install argminer
```
If looking to install from the latest commit, please use the following:
```bash
pip install argminer@git+https://git@github.com/namiyousef/argument-mining.git@develop 
```

## Argminer features

Datasets in the field of argument mining are stored in completely different formats and have different labels. This makes it difficult to compare and contrast model performance on them provided different configurations and processing steps. 

The data processing API provides a standard method of processing these datasets end-to-end to enable fast and consistent experimentation and modelling.

<img width="1665" alt="EndToEnd_bw" src="https://user-images.githubusercontent.com/64047828/163906308-c1fef74e-141c-4e09-8e13-01c8be0ef625.png">

### Labelling Strategies

SOTA argument mining methods treat the process of chunking text into it's argument types and classifying them as an NER problem on long sequences. This means that segment of text with it's associated label is converted into a sequence of classes for the model to predict. To illustrate this we will be using the following passage:

```python
"Natural Language Processing is the best field in Machine Learning.According to a recent poll by DeepLearning.ai it's popularity has increased by twofold in the last 2 years."
```


Let's suppose for the sake of argument that the passage has the following labels:

**Sentence 1: Claim**
```python
sentence_1 = "Natural Language Processing is the best field in Machine Learning."
label_2 = "Claim"
```

**Sentence 2: Evidence**
```python
sentence_2 = "According to a recent poll by DeepLearning.ai it's popularity has increased by twofold in the last 2 years."
label_2 = "Evidence"
```

From this we can create a vector with the associated label for each word. Thus the whole text would have an assoicated label as follows:

```python
labels = ["Claim"]*10 + ["Evidence"]*18  # numbers are length of split sentences
```

With the NER style of labelling, you can modify the above to indicate whether a label is the beginning of a chunk, the end of a chunk or inside a chunk. For example, `"According"` could be labelled as `"B-Evidence"` in sentence 2, and the subsequent parts of it would be labelled as `"I-Evidence"`.

The above is easy if considering a split on words, however this becomes more complicated when we considered how transformer models work. These tokenise inputs based on substrings, and therefore the labels have to be extended. Further questions are raised, for instance: if the word according is split into `"Accord"` and `"ing"`, do we label both of these as `"B-Evidence"` or should we label `"ing"` as `"I-Evidence"`? Or do we just ignore subtokens? Or do we label subtokens as a completely separate class?

We call a labelling strategy that keeps the subtoken labels the same as the start token labels `"wordLevel"` and a strategy that differentiates between them as `"standard"`. Further we provide the following labelling strategies:

- **IO:** Argument tokens are classified as `"I-{arg_type}"` and non-argument tokens as `"O"`
- **BIO:** Argument start tokens are classified as `"B-{arg_type}"`, all other argument tokens as `"I-{arg_type}"`". Non-argument tokens as `"O"`
- **BIEO:** Argument start tokens are classified as `"B-{arg_type}"`, argument end tokens as `"E-{arg_type}"` and all other argument tokens as `"I-{arg_type}"`. Non-argument tokens as `"O"`
- **BIXO:** First argument start tokens are classified as `"B-{arg_type}"`, other argument start tokens are classified as `"I-{arg_type}"`. Argument subtokens are classified as `"X"`. Non-argument tokens as `"O"`

Considering all combinations, the processor API provies functionality for the following strategies:

- standard_io
- wordLevel_io
- standard_bio
- wordLevel_bio
- standard_bieo*
- wordLevel_bieo
- standard_bixo**

* `B-` labels are prioritised over `E-` tokens, e.g. for a single word sentence the word would be labelled as `B-`.

** This method is not one that is backed by literature. It is something that we thought would be interesting to examine. The intuition is that the `X` label captures grammatical elements of argument segments.

### Data Augmentation (Adversarial Examples)

The DataProcessor is designed such that it allows easy

### Evaluation

- TODO

# Quick Start

## Web API

After installing with `pip` create a new virtual environment and activate it:
```bash
python3 -m venv venv
source venv/bin/activate
```

Then run the following command to launch the Web API.
```bash
argminer-api
```

Note that this requires your port `8080` to be free.


## Development Kit

A quick start showing how to use the DataProcessor for the AAE dataset.
```python
from argminer.data import ArgumentMiningDataset, TUDarmstadtProcessor
from argminer.evaluation import inference
from argminer.config import LABELS_MAP_DICT
from torch.utils.data import DataLoader
from torch.optim import Adam
from transformers import AutoModelForTokenClassification, AutoTokenizer

# set path to data source
path = 'ArgumentAnnotatedEssay-2.0'

processor = TUDarmstadtProcessor(path)
processor = processor.preprocess()

# augmenter
def hello_world_augmenter(text):
    text = ['Hello'] + text.split() + ['World']
    text = ' '.join(text)
    return text

processor = processor.process('bieo', processors=[hello_world_augmenter]).postprocess()

df_dict = processor.get_tts(test_size=0.3)
df_train = df_dict['train'][['text', 'labels']]
df_test = df_dict['test'][['text', 'labels']]

df_label_map = LABELS_MAP_DICT['TUDarmstadt']['bieo']

max_length = 1024

# datasets
tokenizer = AutoTokenizer.from_pretrained('google/bigbird-roberta-base', add_prefix_space=True)
model = AutoModelForTokenClassification.from_pretrained('google/bigbird-roberta-base')
optimizer = Adam(model.parameters())

trainset = ArgumentMiningDataset(
    df_label_map, df_train, tokenizer, max_length
)
testset = ArgumentMiningDataset(
    df_label_map, df_train, tokenizer, max_length, is_train=False
)

train_loader = DataLoader(trainset)
test_loader = DataLoader(testset)

# sample training script (very simplistic, see run.py in cluster/cluster_setup/job_files for a full-fledged one)
epochs = 1
for epoch in range(epochs):
    model.train()

    for i, (inputs, targets) in enumerate(train_loader):

        optimizer.zero_grad()

        loss, outputs = model(
            labels=targets,
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            return_dict=False
        )

        # backward pass

        loss.backward()
        optimizer.step()

# run inference
df_metrics, df_scores = inference(model, test_loader)

```



# References
<a id="1">[1]</a>
Christian S. and Iryna G. (2017). _Parsing Argumentation Structures in Persuasive Essays_. DOI: 10.1162/COLI_a_00295

<a id="2">[2]</a> 
Scott C. and the Learning Agency. url: https://github.com/scrosseye/PERSUADE_corpus

<a id="3">[3]</a> 
Alhindi, T. and Ghosh, D. (2021). 
"Sharks are not the threat humans are": _Argument Component Segmentation in School Student Essays_. Proceedings of the 16th Workshop on Innovative Use of NLP for Building Educational Applications. p.210-222. url: https://aclanthology.org/2021.bea-1.22
