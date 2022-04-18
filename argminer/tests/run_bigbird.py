# %% [markdown]
# # PyTorch BigBird NER Baseline - CV 0.615
# This notebook is a PyTorch starter notebook for Kaggle's "Feedback Prize - Evaluating Student Writing" Competition. It demonstrates how to train, infer, and submit a model to Kaggle without internet. Currently this notebook uses
#
# * backbone BigBird  (with HuggingFace's head for TokenClassification)
# * NER formulation (with `is_split_into_words=True` tokenization)
# * one fold
#
# By changing a few lines of code, we can use this notebook to evaluate different PyTorch backbones! And we can run all sorts of other experiments. If we try a backbone that doesn't accept 1024 wide tokens (like BigBird or LongFormer), then we can add a sliding window to train and inference. BigBird is a new SOTA transformer with arXiv paper [here][3] which can accept large token inputs as wide as 4096!
#
# The model in this notebook uses HuggingFace's `AutoModelForTokenClassification`. If we want a custom head, we could use `AutoModel` and then build our own head. See my TensorFlow notebook [here][2] for an example.
#
# The tokenization process uses `tokenizer(txt.split(), is_split_into_words=True)`, note that this ignores characters like `\n`. If we want our model to see new paragraphs, we need to rewrite this code and avoid `is_split_into_words=True`. See my TensorFlow notebook [here][2] for an example.
#
# This notebook uses many code cells from Raghavendrakotala's great notebook [here][1]. Don't forget to upvote Raghavendrakotala's notebook :-)
#
# [1]: https://www.kaggle.com/raghavendrakotala/fine-tunned-on-roberta-base-as-ner-problem-0-533
# [2]: https://www.kaggle.com/cdeotte/tensorflow-longformer-ner-cv-0-617
# [3]: https://arxiv.org/abs/2007.14062

# %% [markdown]
# # Configuration
# This notebook can either train a new model or load a previously trained model (made from previous notebook version). Furthermore, this notebook can either create new NER labels or load existing NER labels (made from previous notebook version). In this notebook version, we will load model and load NER labels.
#
# Also this notebook can load huggingface stuff (like tokenizers) from a Kaggle dataset, or download it from internet. (If it downloads from internet, you can then put it in a Kaggle dataset, so next time you can turn internet off).

# %% [code] {"execution":{"iopub.status.busy":"2022-03-12T18:24:18.036934Z","iopub.execute_input":"2022-03-12T18:24:18.037239Z","iopub.status.idle":"2022-03-12T18:24:18.042773Z","shell.execute_reply.started":"2022-03-12T18:24:18.037207Z","shell.execute_reply":"2022-03-12T18:24:18.041948Z"}}
import os
from os.path import join
import time

ROOT_DIR = 'data'
DATA_DIR = join(ROOT_DIR, 'feedback-prize-2021')
TRAIN_DIR = join(DATA_DIR, 'train')
TEST_DIR = join(DATA_DIR, 'test')
TRAIN_CSV_PATH = join(DATA_DIR, 'train.csv')



# DECLARE HOW MANY GPUS YOU WISH TO USE.
# KAGGLE ONLY HAS 1, BUT OFFLINE, YOU CAN USE MORE
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 0,1,2,3 for four gpu

# VERSION FOR SAVING MODEL WEIGHTS
VER = os.environ.get('VER', 0)
LOC_MODEL_NAME = os.environ.get('LOC_MODEL_NAME', 'py-bigbird')
LOC_MODEL_NAME = f'{LOC_MODEL_NAME}_v{VER}'


# IF VARIABLE IS NONE, THEN NOTEBOOK COMPUTES TOKENS
# OTHERWISE NOTEBOOK LOADS TOKENS FROM PATH
LOAD_TOKENS_FROM = join(ROOT_DIR, LOC_MODEL_NAME) if not VER else None

# IF VARIABLE IS NONE, THEN NOTEBOOK TRAINS A NEW MODEL
# OTHERWISE IT LOADS YOUR PREVIOUSLY TRAINED MODEL
LOAD_MODEL_FROM = join(ROOT_DIR, LOC_MODEL_NAME) if not VER else None

# IF FOLLOWING IS NONE, THEN NOTEBOOK
# USES INTERNET AND DOWNLOADS HUGGINGFACE
# CONFIG, TOKENIZER, AND MODEL
DOWNLOADED_MODEL_PATH = join(ROOT_DIR, LOC_MODEL_NAME) if not VER else None

if DOWNLOADED_MODEL_PATH is None:
    DOWNLOADED_MODEL_PATH = 'model'
MODEL_NAME = 'google/bigbird-roberta-base'

# %% [code] {"execution":{"iopub.status.busy":"2022-03-12T18:24:20.121002Z","iopub.execute_input":"2022-03-12T18:24:20.121557Z","iopub.status.idle":"2022-03-12T18:24:20.130501Z","shell.execute_reply.started":"2022-03-12T18:24:20.121517Z","shell.execute_reply":"2022-03-12T18:24:20.129527Z"}}
from torch import cuda

config = {'model_name': MODEL_NAME,
          'max_length': 1024,
          'train_batch_size': 4,
          'valid_batch_size': 4,
          'epochs': 5,
          'learning_rates': [2.5e-5, 2.5e-5, 2.5e-6, 2.5e-6, 2.5e-7],
          'max_grad_norm': 10,
          'device': 'cuda' if cuda.is_available() else 'cpu'}

# THIS WILL COMPUTE VAL SCORE DURING COMMIT BUT NOT DURING SUBMIT
COMPUTE_VAL_SCORE = True
if len(os.listdir(TEST_DIR)) > 5:
    COMPUTE_VAL_SCORE = False

# %% [markdown]
# # How To Submit PyTorch Without Internet
# Many people ask me, how do I submit PyTorch models without internet? With HuggingFace Transformer, it's easy. Just download the following 3 things (1) model weights, (2) tokenizer files, (3) config file, and upload them to a Kaggle dataset. Below shows code how to get the files from HuggingFace for Google's BigBird-base. But this same code can download any transformer, like for example roberta-base.

# %% [code] {"execution":{"iopub.status.busy":"2022-03-12T18:24:20.852257Z","iopub.execute_input":"2022-03-12T18:24:20.852566Z","iopub.status.idle":"2022-03-12T18:24:20.871892Z","shell.execute_reply.started":"2022-03-12T18:24:20.852529Z","shell.execute_reply":"2022-03-12T18:24:20.871218Z"}}
from transformers import *

if DOWNLOADED_MODEL_PATH == 'model':
    os.mkdir('model')

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)
    tokenizer.save_pretrained('model')

    config_model = AutoConfig.from_pretrained(MODEL_NAME)
    config_model.num_labels = 15
    config_model.save_pretrained('model')

    backbone = AutoModelForTokenClassification.from_pretrained(MODEL_NAME,
                                                               config=config_model)
    backbone.save_pretrained('model')

# %% [markdown]
# # Load Data and Libraries
# In addition to loading the train dataframe, we will load all the train and text files and save them in a dataframe.

# %% [code] {"papermill":{"duration":3.432226,"end_time":"2021-12-21T12:12:07.26275","exception":false,"start_time":"2021-12-21T12:12:03.830524","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-03-12T18:24:24.409513Z","iopub.execute_input":"2022-03-12T18:24:24.409774Z","iopub.status.idle":"2022-03-12T18:24:24.416351Z","shell.execute_reply.started":"2022-03-12T18:24:24.409745Z","shell.execute_reply":"2022-03-12T18:24:24.41557Z"}}
import numpy as np, os
import pandas as pd, gc
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForTokenClassification
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.metrics import accuracy_score

# %% [code] {"papermill":{"duration":1.866495,"end_time":"2021-12-21T12:12:09.158087","exception":false,"start_time":"2021-12-21T12:12:07.291592","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-03-12T18:24:25.052334Z","iopub.execute_input":"2022-03-12T18:24:25.052866Z","iopub.status.idle":"2022-03-12T18:24:25.821754Z","shell.execute_reply.started":"2022-03-12T18:24:25.052828Z","shell.execute_reply":"2022-03-12T18:24:25.820976Z"}}
train_df = pd.read_csv(TRAIN_CSV_PATH)
print(train_df.shape)
train_df.head()

# %% [code] {"papermill":{"duration":0.083228,"end_time":"2021-12-21T12:12:09.396487","exception":false,"start_time":"2021-12-21T12:12:09.313259","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-03-12T18:24:27.970141Z","iopub.execute_input":"2022-03-12T18:24:27.970677Z","iopub.status.idle":"2022-03-12T18:24:27.986326Z","shell.execute_reply.started":"2022-03-12T18:24:27.970642Z","shell.execute_reply":"2022-03-12T18:24:27.985731Z"}}
# https://www.kaggle.com/raghavendrakotala/fine-tunned-on-roberta-base-as-ner-problem-0-533
test_names, test_texts = [], []
for f in list(os.listdir(TEST_DIR)):
    test_names.append(f.replace('.txt', ''))
    test_texts.append(open(os.path.join(TEST_DIR, f), 'r').read())
test_texts = pd.DataFrame({'id': test_names, 'text': test_texts})
test_texts.head()

# %% [code] {"papermill":{"duration":38.695201,"end_time":"2021-12-21T12:12:48.120383","exception":false,"start_time":"2021-12-21T12:12:09.425182","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-03-12T18:24:32.2065Z","iopub.execute_input":"2022-03-12T18:24:32.206949Z","iopub.status.idle":"2022-03-12T18:24:39.084573Z","shell.execute_reply.started":"2022-03-12T18:24:32.206892Z","shell.execute_reply":"2022-03-12T18:24:39.083833Z"}}
# https://www.kaggle.com/raghavendrakotala/fine-tunned-on-roberta-base-as-ner-problem-0-533
test_names, train_texts = [], []

for f in tqdm(list(os.listdir(TRAIN_DIR))):
    test_names.append(f.replace('.txt', ''))
    train_texts.append(open(os.path.join(TRAIN_DIR, f), 'r').read())
train_text_df = pd.DataFrame({'id': test_names, 'text': train_texts})
train_text_df.head()

# %% [markdown] {"papermill":{"duration":0.123678,"end_time":"2021-12-21T12:12:48.368476","exception":false,"start_time":"2021-12-21T12:12:48.244798","status":"completed"},"tags":[]}
# # Convert Train Text to NER Labels
# We will now convert all text words into NER labels and save in a dataframe.

# %% [code] {"execution":{"iopub.status.busy":"2022-03-12T18:24:43.851532Z","iopub.execute_input":"2022-03-12T18:24:43.851819Z","iopub.status.idle":"2022-03-12T18:24:56.454943Z","shell.execute_reply.started":"2022-03-12T18:24:43.851785Z","shell.execute_reply":"2022-03-12T18:24:56.454225Z"}}
if not LOAD_TOKENS_FROM:
    all_entities = []
    for ii, i in enumerate(train_text_df.iterrows()):
        if ii % 100 == 0: print(ii, ', ', end='')
        total = i[1]['text'].split().__len__()
        entities = ["O"] * total
        for j in train_df[train_df['id'] == i[1]['id']].iterrows():
            discourse = j[1]['discourse_type']
            list_ix = [int(x) for x in j[1]['predictionstring'].split(' ')]
            entities[list_ix[0]] = f"B-{discourse}"
            for k in list_ix[1:]: entities[k] = f"I-{discourse}"
        all_entities.append(entities)
    train_text_df['entities'] = all_entities
    train_text_df.to_csv('train_NER.csv', index=False)

else:
    from ast import literal_eval

    train_text_df = pd.read_csv(f'{LOAD_TOKENS_FROM}/train_NER.csv')
    # pandas saves lists as string, we must convert back
    train_text_df.entities = train_text_df.entities.apply(lambda x: literal_eval(x))

print(train_text_df.shape)
train_text_df.head()

# %% [code] {"papermill":{"duration":0.940609,"end_time":"2021-12-21T12:18:50.456125","exception":false,"start_time":"2021-12-21T12:18:49.515516","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-03-12T18:25:06.69618Z","iopub.execute_input":"2022-03-12T18:25:06.696457Z","iopub.status.idle":"2022-03-12T18:25:06.702495Z","shell.execute_reply.started":"2022-03-12T18:25:06.696427Z","shell.execute_reply":"2022-03-12T18:25:06.701576Z"}}
# CREATE DICTIONARIES THAT WE CAN USE DURING TRAIN AND INFER
output_labels = ['O', 'B-Lead', 'I-Lead', 'B-Position', 'I-Position', 'B-Claim', 'I-Claim', 'B-Counterclaim',
                 'I-Counterclaim',
                 'B-Rebuttal', 'I-Rebuttal', 'B-Evidence', 'I-Evidence', 'B-Concluding Statement',
                 'I-Concluding Statement']

labels_to_ids = {v: k for k, v in enumerate(output_labels)}
ids_to_labels = {k: v for k, v in enumerate(output_labels)}

# %% [code] {"papermill":{"duration":0.994404,"end_time":"2021-12-21T12:18:52.798977","exception":false,"start_time":"2021-12-21T12:18:51.804573","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-03-12T18:25:07.810776Z","iopub.execute_input":"2022-03-12T18:25:07.811647Z","iopub.status.idle":"2022-03-12T18:25:07.817875Z","shell.execute_reply.started":"2022-03-12T18:25:07.811599Z","shell.execute_reply":"2022-03-12T18:25:07.817188Z"}}
labels_to_ids

# %% [markdown] {"papermill":{"duration":1.001889,"end_time":"2021-12-21T12:18:54.981896","exception":false,"start_time":"2021-12-21T12:18:53.980007","status":"completed"},"tags":[]}
# # Define the dataset function
# Below is our PyTorch dataset function. It always outputs tokens and attention. During training it also provides labels. And during inference it also provides word ids to help convert token predictions into word predictions.
#
# Note that we use `text.split()` and `is_split_into_words=True` when we convert train text to labeled train tokens. This is how the HugglingFace tutorial does it. However, this removes characters like `\n` new paragraph. If you want your model to see new paragraphs, then we need to map words to tokens ourselves using `return_offsets_mapping=True`. See my TensorFlow notebook [here][1] for an example.
#
# Some of the following code comes from the example at HuggingFace [here][2]. However I think the code at that link is wrong. The HuggingFace original code is [here][3]. With the flag `LABEL_ALL` we can either label just the first subword token (when one word has more than one subword token). Or we can label all the subword tokens (with the word's label). In this notebook version, we label all the tokens. There is a Kaggle discussion [here][4]
#
# [1]: https://www.kaggle.com/cdeotte/tensorflow-longformer-ner-cv-0-617
# [2]: https://huggingface.co/docs/transformers/custom_datasets#tok_ner
# [3]: https://github.com/huggingface/transformers/blob/86b40073e9aee6959c8c85fcba89e47b432c4f4d/examples/pytorch/token-classification/run_ner.py#L371
# [4]: https://www.kaggle.com/c/feedback-prize-2021/discussion/296713

# %% [code] {"papermill":{"duration":0.934726,"end_time":"2021-12-21T12:18:56.852259","exception":false,"start_time":"2021-12-21T12:18:55.917533","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-03-12T18:25:09.943353Z","iopub.execute_input":"2022-03-12T18:25:09.943718Z","iopub.status.idle":"2022-03-12T18:25:09.954748Z","shell.execute_reply.started":"2022-03-12T18:25:09.943684Z","shell.execute_reply":"2022-03-12T18:25:09.953867Z"}}
LABEL_ALL_SUBTOKENS = True


class dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, get_wids):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.get_wids = get_wids  # for validation

    def __getitem__(self, index):
        # GET TEXT AND WORD LABELS
        text = self.data.text[index]
        word_labels = self.data.entities[index] if not self.get_wids else None

        # TOKENIZE TEXT
        encoding = self.tokenizer(text.split(),
                                  is_split_into_words=True,
                                  # return_offsets_mapping=True,
                                  padding='max_length',
                                  truncation=True,
                                  max_length=self.max_len)
        word_ids = encoding.word_ids()

        # CREATE TARGETS
        if not self.get_wids:
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(labels_to_ids[word_labels[word_idx]])
                else:
                    if LABEL_ALL_SUBTOKENS:
                        label_ids.append(labels_to_ids[word_labels[word_idx]])
                    else:
                        label_ids.append(-100)
                previous_word_idx = word_idx
            encoding['labels'] = label_ids

        # CONVERT TO TORCH TENSORS
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        if self.get_wids:
            word_ids2 = [w if w is not None else -1 for w in word_ids]
            item['wids'] = torch.as_tensor(word_ids2)

        return item

    def __len__(self):
        return self.len


# %% [markdown] {"papermill":{"duration":0.936225,"end_time":"2021-12-21T12:19:08.206923","exception":false,"start_time":"2021-12-21T12:19:07.270698","status":"completed"},"tags":[]}
# # Create Train and Validation Dataloaders
# We will use the same train and validation subsets as my TensorFlow notebook [here][1]. Then we can compare results. And/or experiment with ensembling the validation fold predictions.
#
# [1]: https://www.kaggle.com/cdeotte/tensorflow-longformer-ner-cv-0-617

# %% [code] {"execution":{"iopub.status.busy":"2022-03-12T18:25:11.267476Z","iopub.execute_input":"2022-03-12T18:25:11.267727Z","iopub.status.idle":"2022-03-12T18:25:11.289536Z","shell.execute_reply.started":"2022-03-12T18:25:11.267698Z","shell.execute_reply":"2022-03-12T18:25:11.288817Z"}}
# CHOOSE VALIDATION INDEXES (that match my TF notebook)
IDS = train_df.id.unique()
print('There are', len(IDS), 'train texts. We will split 90% 10% for validation.')

# TRAIN VALID SPLIT 90% 10%
np.random.seed(42)
train_idx = np.random.choice(np.arange(len(IDS)), int(0.9 * len(IDS)), replace=False)
valid_idx = np.setdiff1d(np.arange(len(IDS)), train_idx)
np.random.seed(None)

# %% [code] {"papermill":{"duration":0.953973,"end_time":"2021-12-21T12:19:10.088215","exception":false,"start_time":"2021-12-21T12:19:09.134242","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-03-12T18:25:12.379503Z","iopub.execute_input":"2022-03-12T18:25:12.380039Z","iopub.status.idle":"2022-03-12T18:25:12.484832Z","shell.execute_reply.started":"2022-03-12T18:25:12.380001Z","shell.execute_reply":"2022-03-12T18:25:12.484039Z"}}
# CREATE TRAIN SUBSET AND VALID SUBSET
data = train_text_df[['id', 'text', 'entities']]
train_dataset = data.loc[data['id'].isin(IDS[train_idx]), ['text', 'entities']].reset_index(drop=True)
test_dataset = data.loc[data['id'].isin(IDS[valid_idx])].reset_index(drop=True)

print("FULL Dataset: {}".format(data.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))

tokenizer = AutoTokenizer.from_pretrained(DOWNLOADED_MODEL_PATH)
training_set = dataset(train_dataset, tokenizer, config['max_length'], False)
testing_set = dataset(test_dataset, tokenizer, config['max_length'], True)

# %% [code] {"papermill":{"duration":0.955464,"end_time":"2021-12-21T12:19:12.022567","exception":false,"start_time":"2021-12-21T12:19:11.067103","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-03-12T18:25:13.961431Z","iopub.execute_input":"2022-03-12T18:25:13.961987Z","iopub.status.idle":"2022-03-12T18:25:13.996222Z","shell.execute_reply.started":"2022-03-12T18:25:13.961949Z","shell.execute_reply":"2022-03-12T18:25:13.995334Z"}}
# TRAIN DATASET AND VALID DATASET
train_params = {'batch_size': config['train_batch_size'],
                'shuffle': True,
                'num_workers': 2,
                'pin_memory': True
                }

test_params = {'batch_size': config['valid_batch_size'],
               'shuffle': False,
               'num_workers': 2,
               'pin_memory': True
               }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)

# TEST DATASET
test_texts_set = dataset(test_texts, tokenizer, config['max_length'], True)
test_texts_loader = DataLoader(test_texts_set, **test_params)


# %% [markdown]
# # Train Model
# The PyTorch train function is taken from Raghavendrakotala's great notebook [here][1]. I assume it uses a masked loss which avoids computing loss when target is `-100`. If not, we need to update this.
#
# In Kaggle notebooks, we will train our model for 5 epochs `batch_size=4` with Adam optimizer and learning rates `LR = [2.5e-5, 2.5e-5, 2.5e-6, 2.5e-6, 2.5e-7]`. The loaded model was trained offline with `batch_size=8` and `LR = [5e-5, 5e-5, 5e-6, 5e-6, 5e-7]`. (Note the learning rate changes `e-5`, `e-6`, and `e-7`). Using `batch_size=4` will probably achieve a better validation score than `batch_size=8`, but I haven't tried yet.
#
# [1]: https://www.kaggle.com/raghavendrakotala/fine-tunned-on-roberta-base-as-ner-problem-0-533

# %% [code] {"papermill":{"duration":1.00345,"end_time":"2021-12-21T12:19:31.294225","exception":false,"start_time":"2021-12-21T12:19:30.290775","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2022-03-12T18:25:15.686455Z","iopub.execute_input":"2022-03-12T18:25:15.686705Z","iopub.status.idle":"2022-03-12T18:25:15.701769Z","shell.execute_reply.started":"2022-03-12T18:25:15.686677Z","shell.execute_reply":"2022-03-12T18:25:15.700843Z"}}
# https://www.kaggle.com/raghavendrakotala/fine-tunned-on-roberta-base-as-ner-problem-0-533
def train(epoch):
    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    # tr_preds, tr_labels = [], []

    # put model in training mode
    model.train()

    start_batch_load = time.time()
    for idx, batch in enumerate(training_loader):
        start_train = time.time()

        ids = batch['input_ids'].to(config['device'], dtype=torch.long)
        mask = batch['attention_mask'].to(config['device'], dtype=torch.long)
        labels = batch['labels'].to(config['device'], dtype=torch.long)

        loss, tr_logits = model(input_ids=ids, attention_mask=mask, labels=labels,
                                return_dict=False)
        tr_loss += loss.item()

        nb_tr_steps += 1
        nb_tr_examples += labels.size(0)

        if idx % 200 == 0:
            loss_step = tr_loss / nb_tr_steps
            print(f"Training loss after {idx:04d} training steps: {loss_step}")

        # compute training accuracy
        flattened_targets = labels.view(-1)  # shape (batch_size * seq_len,)
        active_logits = tr_logits.view(-1, model.num_labels)  # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1)  # shape (batch_size * seq_len,)

        # only compute accuracy at active labels
        active_accuracy = labels.view(-1) != -100  # shape (batch_size, seq_len)
        # active_labels = torch.where(active_accuracy, labels.view(-1), torch.tensor(-100).type_as(labels))

        labels = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)

        # tr_labels.extend(labels)
        # tr_preds.extend(predictions)

        tmp_tr_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
        tr_accuracy += tmp_tr_accuracy

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(), max_norm=config['max_grad_norm']
        )

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        end_train = time.time()
        print(
            f'Completed batch {idx + 1}: Total time ({end_train - start_batch_load: .3g}), Load time ({start_train - start_batch_load}), Train time ({end_train - start_train})')
        start_batch_load = time.time()

    epoch_loss = tr_loss / nb_tr_steps
    tr_accuracy = tr_accuracy / nb_tr_steps
    print(f"Training loss epoch: {epoch_loss}")
    print(f"Training accuracy epoch: {tr_accuracy}")


# %% [code] {"execution":{"iopub.status.busy":"2022-03-12T18:25:16.880842Z","iopub.execute_input":"2022-03-12T18:25:16.881533Z","iopub.status.idle":"2022-03-12T18:25:18.454143Z","shell.execute_reply.started":"2022-03-12T18:25:16.881497Z","shell.execute_reply":"2022-03-12T18:25:18.453197Z"}}
# CREATE MODEL
config_model = AutoConfig.from_pretrained(DOWNLOADED_MODEL_PATH + '/config.json')
model = AutoModelForTokenClassification.from_pretrained(
    DOWNLOADED_MODEL_PATH + '/pytorch_model.bin', config=config_model)
model.to(config['device'])
optimizer = torch.optim.Adam(params=model.parameters(), lr=config['learning_rates'][0])

# %% [code] {"execution":{"iopub.status.busy":"2022-03-12T18:22:47.774277Z","iopub.execute_input":"2022-03-12T18:22:47.774961Z","iopub.status.idle":"2022-03-12T18:22:48.120368Z","shell.execute_reply.started":"2022-03-12T18:22:47.774917Z","shell.execute_reply":"2022-03-12T18:22:48.119453Z"}}
# LOOP TO TRAIN MODEL (or load model)
if not LOAD_MODEL_FROM:
    for epoch in range(config['epochs']):

        print(f"### Training epoch: {epoch + 1}")
        for g in optimizer.param_groups:
            g['lr'] = config['learning_rates'][epoch]
        lr = optimizer.param_groups[0]['lr']
        print(f'### LR = {lr}\n')

        train(epoch)
        torch.cuda.empty_cache()
        gc.collect()

    torch.save(model.state_dict(), f'{LOC_MODEL_NAME}_v{VER}.pt')
else:

    model.load_state_dict(torch.load(f'{LOAD_MODEL_FROM}/{LOC_MODEL_NAME}_v{VER}.pt'))
    print('Model loaded.')

# %% [code] {"execution":{"iopub.status.busy":"2022-03-12T18:25:22.54889Z","iopub.execute_input":"2022-03-12T18:25:22.549597Z"}}
for epoch in range(config['epochs']):

    print(f"### Training epoch: {epoch + 1}")
    for g in optimizer.param_groups:
        g['lr'] = config['learning_rates'][epoch]
    lr = optimizer.param_groups[0]['lr']
    print(f'### LR = {lr}\n')

    train(epoch)
    torch.cuda.empty_cache()
    gc.collect()

torch.save(model.state_dict(), f'{LOC_MODEL_NAME}_v{VER}.pt')


# %% [markdown]
# # Inference and Validation Code
# We will infer in batches using our data loader which is faster than inferring one text at a time with a for-loop. The metric code is taken from Rob Mulla's great notebook [here][2]. Our model achieves validation F1 score 0.615!
#
# During inference our model will make predictions for each subword token. Some single words consist of multiple subword tokens. In the code below, we use a word's first subword token prediction as the label for the entire word. We can try other approaches, like averaging all subword predictions or taking `B` labels before `I` labels etc.
#
# [1]: https://www.kaggle.com/raghavendrakotala/fine-tunned-on-roberta-base-as-ner-problem-0-533
# [2]: https://www.kaggle.com/robikscube/student-writing-competition-twitch

# %% [code] {"execution":{"iopub.status.busy":"2021-12-24T18:51:26.626644Z","iopub.execute_input":"2021-12-24T18:51:26.626902Z","iopub.status.idle":"2021-12-24T18:51:26.63465Z","shell.execute_reply.started":"2021-12-24T18:51:26.626869Z","shell.execute_reply":"2021-12-24T18:51:26.633853Z"}}
def inference(batch):
    # MOVE BATCH TO GPU AND INFER
    ids = batch["input_ids"].to(config['device'])
    mask = batch["attention_mask"].to(config['device'])
    outputs = model(ids, attention_mask=mask, return_dict=False)
    all_preds = torch.argmax(outputs[0], axis=-1).cpu().numpy()

    # INTERATE THROUGH EACH TEXT AND GET PRED
    predictions = []
    for k, text_preds in enumerate(all_preds):
        token_preds = [ids_to_labels[i] for i in text_preds]

        prediction = []
        word_ids = batch['wids'][k].numpy()
        previous_word_idx = -1
        for idx, word_idx in enumerate(word_ids):
            if word_idx == -1:
                pass
            elif word_idx != previous_word_idx:
                prediction.append(token_preds[idx])
                previous_word_idx = word_idx
        predictions.append(prediction)

    return predictions


# %% [code] {"execution":{"iopub.status.busy":"2021-12-24T18:51:26.636067Z","iopub.execute_input":"2021-12-24T18:51:26.636584Z","iopub.status.idle":"2021-12-24T18:51:26.647585Z","shell.execute_reply.started":"2021-12-24T18:51:26.636546Z","shell.execute_reply":"2021-12-24T18:51:26.646928Z"}}
# https://www.kaggle.com/zzy990106/pytorch-ner-infer
# code has been modified from original
def get_predictions(df=test_dataset, loader=testing_loader):
    # put model in training mode
    model.eval()

    # GET WORD LABEL PREDICTIONS
    y_pred2 = []
    for batch in loader:
        labels = inference(batch)
        y_pred2.extend(labels)

    final_preds2 = []
    for i in range(len(df)):

        idx = df.id.values[i]
        # pred = [x.replace('B-','').replace('I-','') for x in y_pred2[i]]
        pred = y_pred2[i]  # Leave "B" and "I"
        preds = []
        j = 0
        while j < len(pred):
            cls = pred[j]
            if cls == 'O':
                j += 1
            else:
                cls = cls.replace('B', 'I')  # spans start with B
            end = j + 1
            while end < len(pred) and pred[end] == cls:
                end += 1

            if cls != 'O' and cls != '' and end - j > 7:
                final_preds2.append((idx, cls.replace('I-', ''),
                                     ' '.join(map(str, list(range(j, end))))))

            j = end

    oof = pd.DataFrame(final_preds2)
    oof.columns = ['id', 'class', 'predictionstring']

    return oof


# %% [code] {"execution":{"iopub.status.busy":"2021-12-24T18:51:26.649284Z","iopub.execute_input":"2021-12-24T18:51:26.649674Z","iopub.status.idle":"2021-12-24T18:51:26.66552Z","shell.execute_reply.started":"2021-12-24T18:51:26.649639Z","shell.execute_reply":"2021-12-24T18:51:26.664844Z"}}
# from Rob Mulla @robikscube
# https://www.kaggle.com/robikscube/student-writing-competition-twitch
def calc_overlap(row):
    """
    Calculates the overlap between prediction and
    ground truth and overlap percentages used for determining
    true positives.
    """
    set_pred = set(row.predictionstring_pred.split(' '))
    set_gt = set(row.predictionstring_gt.split(' '))
    # Length of each and intersection
    len_gt = len(set_gt)
    len_pred = len(set_pred)
    inter = len(set_gt.intersection(set_pred))
    overlap_1 = inter / len_gt
    overlap_2 = inter / len_pred
    return [overlap_1, overlap_2]


def score_feedback_comp(pred_df, gt_df):
    """
    A function that scores for the kaggle
        Student Writing Competition

    Uses the steps in the evaluation page here:
        https://www.kaggle.com/c/feedback-prize-2021/overview/evaluation
    """
    gt_df = gt_df[['id', 'discourse_type', 'predictionstring']] \
        .reset_index(drop=True).copy()
    pred_df = pred_df[['id', 'class', 'predictionstring']] \
        .reset_index(drop=True).copy()
    pred_df['pred_id'] = pred_df.index
    gt_df['gt_id'] = gt_df.index
    # Step 1. all ground truths and predictions for a given class are compared.
    joined = pred_df.merge(gt_df,
                           left_on=['id', 'class'],
                           right_on=['id', 'discourse_type'],
                           how='outer',
                           suffixes=('_pred', '_gt')
                           )
    joined['predictionstring_gt'] = joined['predictionstring_gt'].fillna(' ')
    joined['predictionstring_pred'] = joined['predictionstring_pred'].fillna(' ')

    joined['overlaps'] = joined.apply(calc_overlap, axis=1)

    # 2. If the overlap between the ground truth and prediction is >= 0.5,
    # and the overlap between the prediction and the ground truth >= 0.5,
    # the prediction is a match and considered a true positive.
    # If multiple matches exist, the match with the highest pair of overlaps is taken.
    joined['overlap1'] = joined['overlaps'].apply(lambda x: eval(str(x))[0])
    joined['overlap2'] = joined['overlaps'].apply(lambda x: eval(str(x))[1])

    joined['potential_TP'] = (joined['overlap1'] >= 0.5) & (joined['overlap2'] >= 0.5)
    joined['max_overlap'] = joined[['overlap1', 'overlap2']].max(axis=1)
    tp_pred_ids = joined.query('potential_TP') \
        .sort_values('max_overlap', ascending=False) \
        .groupby(['id', 'predictionstring_gt']).first()['pred_id'].values

    # 3. Any unmatched ground truths are false negatives
    # and any unmatched predictions are false positives.
    fp_pred_ids = [p for p in joined['pred_id'].unique() if p not in tp_pred_ids]

    matched_gt_ids = joined.query('potential_TP')['gt_id'].unique()
    unmatched_gt_ids = [c for c in joined['gt_id'].unique() if c not in matched_gt_ids]

    # Get numbers of each type
    TP = len(tp_pred_ids)
    FP = len(fp_pred_ids)
    FN = len(unmatched_gt_ids)
    # calc microf1
    my_f1_score = TP / (TP + 0.5 * (FP + FN))
    return my_f1_score


# %% [code] {"execution":{"iopub.status.busy":"2021-12-24T18:51:26.66668Z","iopub.execute_input":"2021-12-24T18:51:26.666937Z","iopub.status.idle":"2021-12-24T18:53:12.478642Z","shell.execute_reply.started":"2021-12-24T18:51:26.666902Z","shell.execute_reply":"2021-12-24T18:53:12.477793Z"}}
if COMPUTE_VAL_SCORE:  # note this doesn't run during submit
    # VALID TARGETS
    valid = train_df.loc[train_df['id'].isin(IDS[valid_idx])]

    # OOF PREDICTIONS
    oof = get_predictions(test_dataset, testing_loader)

    # COMPUTE F1 SCORE
    f1s = []
    CLASSES = oof['class'].unique()
    for c in CLASSES:
        pred_df = oof.loc[oof['class'] == c].copy()
        gt_df = valid.loc[valid['discourse_type'] == c].copy()
        f1 = score_feedback_comp(pred_df, gt_df)
        print(c, f1)
        f1s.append(f1)
    print('Overall', np.mean(f1s))

# %% [markdown] {"papermill":{"duration":1.170872,"end_time":"2021-12-21T12:58:34.316729","exception":false,"start_time":"2021-12-21T12:58:33.145857","status":"completed"},"tags":[]}
# # Infer Test Data and Write Submission CSV
# We will now infer the test data and write submission CSV

# %% [code] {"papermill":{"duration":0.998396,"end_time":"2021-12-21T12:58:50.260737","exception":false,"start_time":"2021-12-21T12:58:49.262341","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-12-24T18:53:12.480154Z","iopub.execute_input":"2021-12-24T18:53:12.48045Z","iopub.status.idle":"2021-12-24T18:53:13.074441Z","shell.execute_reply.started":"2021-12-24T18:53:12.480399Z","shell.execute_reply":"2021-12-24T18:53:13.073708Z"}}
sub = get_predictions(test_texts, test_texts_loader)
sub.head()

# %% [code] {"papermill":{"duration":1.020359,"end_time":"2021-12-21T12:58:54.413788","exception":false,"start_time":"2021-12-21T12:58:53.393429","status":"completed"},"tags":[],"execution":{"iopub.status.busy":"2021-12-24T18:53:13.076066Z","iopub.execute_input":"2021-12-24T18:53:13.076315Z","iopub.status.idle":"2021-12-24T18:53:13.083963Z","shell.execute_reply.started":"2021-12-24T18:53:13.07628Z","shell.execute_reply":"2021-12-24T18:53:13.083318Z"}}
sub.to_csv("submission.csv", index=False)