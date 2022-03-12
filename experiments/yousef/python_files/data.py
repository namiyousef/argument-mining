from torch.utils.data import DataLoader, Dataset
import os
import time
import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype, is_string_dtype
import torch

class DataProcessor:
    """
    Class for standardising and processing data from raw form
    """
    def __init__(self, docs):
        pass

    def preprocess(self):
        """ This is pretokenisation cleaning"""
        pass

    def process(self):
        """ This is usually the tokeniser"""
        pass

    def postprocess(self):
        """ This is post tokenisation cleaning"""
        pass

def generate_class_to_ent_map(unique_classes):
    # sort classes
    pass

def df_from_text_files(path_to_dir):
    filenames = [filename for filename in os.listdir(path_to_dir)]
    records = [(filename.rstrip('.txt'), open(os.path.join(path_to_dir, filename), 'r').read()) for filename in filenames]
    df = pd.DataFrame.from_records(records, columns=['id', 'text'])
    return df

def create_labels_doc_level(path_to_text_dir, path_to_ground_truth):
    s = time.time()
    df_ground_truth = pd.read_csv(path_to_ground_truth)
    unique_labels = list(df_ground_truth.discourse_type.unique())
    unique_labels = [f'{start_letter}-{label}' for label in unique_labels for start_letter in ['B', 'I']]
    label_to_id_map = {
        label: i for i, label in enumerate(
            ['O'] + unique_labels
        )
    }

    df_ground_truth.predictionstring = df_ground_truth.predictionstring.str.split()
    df_ground_truth['label_ids'] = df_ground_truth.predictionstring.apply(lambda x: [int(x[0]), int(x[-1])])
    df_ground_truth['labels'] = df_ground_truth[['discourse_type', 'label_ids']].apply(
        lambda x: [f'B-{x.discourse_type}'] + [f'I-{x.discourse_type}']*(x.label_ids[-1] - x.label_ids[0]),
        axis=1
    )
    df_ground_truth['range'] = df_ground_truth.label_ids.apply(lambda x: np.arange(x[0], x[1]+1))
    df_ground_truth['labels'] = df_ground_truth.labels.apply(lambda x: [label_to_id_map[label] for label in x])

    # this is kind of wrong?

    df_texts = df_from_text_files(path_to_text_dir)
    """df_texts.text = df_texts.text.str.replace('\n', ' ')
    df_texts.text = df_texts.text.str.replace('\s+', ' ')
    df_texts.text = df_texts.text.str.replace('(?<=\w) (?=[.,\/#!$%\^&\*;:{}=\-_`~()])', '')"""
    df_texts.text = df_texts.text.str.strip()
    df_texts['text_split'] = df_texts.text.str.split()
    df_texts['labels'] = df_texts.text_split.apply(lambda x: len(x)*[label_to_id_map['O']])
    df_texts = df_texts.merge(
        df_ground_truth.groupby('id').agg({
            'range': lambda x: np.concatenate(list(x)),
            'labels': lambda x: np.concatenate(list(x))
        }).rename(columns={'labels':'labels_temp'}),
        on='id'
    )
    def update_inplace(x):
        ids = x.range
        new_labels = x.labels_temp
        labels = np.array(x.labels, dtype=new_labels.dtype)
        assert len(ids) == len(new_labels)
        labels[ids] = new_labels
        return list(labels)


    df_texts.labels = df_texts.apply(lambda x: update_inplace(x), axis=1)

    """df_texts.text = df_texts.text.str.replace('\n', ' ')
    df_texts.text = df_texts.text.str.replace('\s+', ' ')
    df_texts.text = df_texts.text.str.replace('(?<=\w) (?=[.,\/#!$%\^&\*;:{}=\-_`~()])', '')"""

    #for i, (text, label) in df_texts[['text', 'labels']].iterrows():
    #    assert text.split().__len__() == label.__len__(), f'Failed because of size mismatch on id: {i}. Shape mismatch {text.split().__len__(), label.__len__()}'


    """ids = np.stack(df_texts.range)
    print(ids.shape)
    labels = np.stack(df_texts.labels)
    print(labels.shape)
    new_labels = np.stack(df_texts.labels)
    print(new_labels.shape)

    labels[ids] = new_labels

    df_texts.labels = labels"""

    return df_texts



if __name__ == '__main__':
    """PATH = '../../../data/kaggle/feedback-prize-2021/train'
    FILE_PATH = '../../../data/kaggle/feedback-prize-2021/train.csv'

    df_texts = create_labels_doc_level(PATH, FILE_PATH)
    df_texts[['text', 'labels']].to_csv('../../../data/kaggle/df_cleaned.csv')"""

    PATH = '../../../data/kaggle/df_cleaned.csv'
    df = pd.read_csv(PATH, index_col=0)
    df.labels = df.labels.apply(lambda x: [int(x_) for x_ in x[1:-1].split(', ')])

    def extend_labels_based_on_tokenizer(words, labels, tokenizer):
        tokens = []
        new_labels = []

        for word, label in zip(words.split(), labels):
            tokenized_word = tokenizer.tokenize(word)
            l_tokenized = len(tokenized_word)
            tokens += tokenized_word
            new_labels += [label] * l_tokenized

        return tokens, new_labels



    from transformers import BigBirdTokenizer

    tokenizer = BigBirdTokenizer.from_pretrained('google/bigbird-roberta-large')



    sentence = 'Yousef Nami is from Iran'
    labels = 'B-PERS I-PERS O O B-Place'
    sentence = '_You sef _Na mi is from _Ir an'
    labels = 'B_PERS B-PERS I-PERS I-PERS'