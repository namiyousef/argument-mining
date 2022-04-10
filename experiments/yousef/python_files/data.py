from torch.utils.data import DataLoader, Dataset
import os
import time
import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype, is_string_dtype
import torch

def _first_appearance_of_unique_item(x):
    """ Torch function to get the first appearance of a unique item"""
    unique, inverse = torch.unique(x, sorted=True, return_inverse=True)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    perm = inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)
    return perm

class ArgumentMiningDataset(Dataset):
    """
    Class for loading data in batches and processing it
    """
    def __init__(self, df_label_map, df_text, tokenizer, max_length, strategy):
        super().__init__()

        assert sorted(df_text.columns) == ['labels', 'text'], f"Please make sure input dataframe has the columns (text, labels)"
        assert 'O' in df_label_map.label.values, 'You are missing label "O"'
        assert strategy in ['standard_bio', 'word_level_bio', 'standard_bixo', 'standard_bieo', 'word_level_bieo'], 'Please use a valid strategy'

        strategy_level, strategy_name = strategy.split('_')

        self.strategy_level = strategy_level
        self.strategy_name = strategy_name

        if strategy_name == 'bio':
            labels = df_label_map.label[df_label_map.label.str.contains('-')].apply(lambda x: x[:1])
            labels = list(labels.unique())
            assert 'BI' == ''.join(sorted(labels)), 'You are missing one of labels "B" or "I"'
        elif strategy_name == 'bieo':
            labels = df_label_map.label[df_label_map.label.str.contains('-')].apply(lambda x: x[:1])
            labels = list(labels.unique())
            assert 'BIE' == ''.join(sorted(labels)), 'You are missing one of labels "B", "I" or "E"'
        elif strategy_name == 'bixo':
            assert 'X' in df_label_map.label.values, 'You are missing label "X"'
            labels = df_label_map.label[df_label_map.label.str.contains('-')].apply(lambda x: x[:1])
            labels = list(labels.unique())
            assert 'BI' == ''.join(sorted(labels)), 'You are missing one of labels "B" or "I"'
        else:
            raise NotImplementedError(f'Support for labelling strategy {strategy} does not exist yet')


        self.inputs = df_text.text.values
        if not is_string_dtype(self.inputs): raise TypeError('Text data must be string type')

        self.consider_end = 'e' in strategy_name # TODO make more robust
        self.use_x = 'x' in strategy_name # TODO make more robust

        # create hash table
        self.label_to_id = {
            label: id_ for label, id_ in df_label_map[['label', 'label_id']].values
        }
        self.id_to_label = {
            id_: label for label, id_ in self.label_to_id.items()
        }
        self.targets = df_text.labels.apply(
            lambda x: [self.label_to_id[label] for label in x]
        ).values



        # -- prepare tokenizer
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        # self.inputs anf self.targets must be of a type that is indexible as shown
        inputs = self.inputs[index]
        targets = self.targets[index]

        # TODO this is for the sentencepiece tokenizer, might have to change for wordpiece
        inputs = self.tokenizer(
            # consider parametrising these
            inputs.split(),
            is_split_into_words=True,  # this means that extra \n should be ignored
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        )
        word_ids = inputs.word_ids()
        word_id_mask = [word_id is not None for word_id in word_ids]
        # TODO maybe this should go in the function?
        word_ids = [word_id for word_id in word_ids if word_id is not None]

        targets = torch.as_tensor(targets, dtype=torch.long)

        labeller = getattr(self, f'_label_{self.strategy_level}')
        targets = labeller(targets, word_id_mask, word_ids)

        """# TODO need to think about sending things to the GPU, which ones to send
        inputs = {
            key: torch.as_tensor(val, dtype=torch.long) for key, val in inputs.items()
        }
        inputs['doc_ids'] = index  # TODO may have to convert to type long
        inputs['word_ids'] = word_ids

        targets = torch.as_tensor(targets, dtype=torch.long)
        expanded_targets = torch.zeros(self.max_length, dtype=torch.long)
        expanded_targets[word_id_mask] = targets[word_ids]"""

        return (inputs, targets)


    def _label_word_level(self, targets, word_id_mask, word_ids):
        expanded_targets = torch.zeros(self.max_length, dtype=torch.long)
        expanded_targets[word_id_mask] = targets[word_ids]
        return expanded_targets

    def _label_standard(self, targets, word_id_mask, word_ids):
        expanded_targets = torch.zeros(self.max_length, dtype=torch.long)
        expanded_targets[word_id_mask] = targets[word_ids]
        word_start_ids = _first_appearance_of_unique_item(torch.as_tensor(word_ids))
        unique_word_ids, word_id_counts = torch.unique(torch.as_tensor(word_ids), return_counts=True)

        # here define the start and end labels
        for i, (word_start_id, word_id, word_id_count) in enumerate(
                zip(word_start_ids, unique_word_ids, word_id_counts)):
            curr_target = expanded_targets[word_start_id].item()
            if curr_target:  # step to filter the orhers
                if word_id_count > 1:
                    ids = list(range(word_start_id, word_start_id + word_id_count))

                    # TODO can make robust by adding string condition 'E-'
                    position, target_label = self.id_to_label[curr_target].split('-')
                    if self.consider_end and 'E' == position:
                        ids = ids[:-1]
                    else:
                        ids = ids[1:]

                    expanded_target_label = self.label_to_id['X'] if self.use_x else self.label_to_id[f'I-{target_label}']
                    expanded_targets[ids] = expanded_target_label # this label needs to be changed!
        return expanded_targets
class KaggleDataset(Dataset):
    """
    Class for loading data in batches after it has been processed
    """
    def __init__(self, dataframe, tokenizer, max_length):

        super().__init__()

        # -- prepare data
        assert sorted(dataframe.columns) == ['labels', 'text'], f"Please make sure input dataframe has the columns (text, labels)"
        # data must be in the correct format
        self.inputs = dataframe.text.values
        self.targets = dataframe.labels.values
        if not is_string_dtype(self.inputs): raise TypeError('Text data must be string type')
        # TODO assertion below is bug; not deleting so remember to add correct assertions
        #if not is_integer_dtype(self.targets): raise TypeError('Label data must be integer type')

        # -- prepare tokenizer
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        # self.inputs anf self.targets must be of a type that is indexible as shown
        inputs = self.inputs[index]
        targets = self.targets[index]

        inputs = self.tokenizer(
            # consider parametrising these
            inputs.split(),
            is_split_into_words=True, # this means that extra \n should be ignored
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        )

        word_ids = inputs.word_ids()
        word_id_mask = [word_id is not None for word_id in word_ids]
        word_ids = [word_id for word_id in word_ids if word_id is not None]

        # TODO need to think about sending things to the GPU, which ones to send
        inputs = {
            key: torch.as_tensor(val, dtype=torch.long) for key, val in inputs.items()
        }
        inputs['doc_ids'] = index # TODO may have to convert to type long
        inputs['word_ids'] = word_ids

        targets = torch.as_tensor(targets, dtype=torch.long)
        expanded_targets = torch.zeros(self.max_length, dtype=torch.long)
        expanded_targets[word_id_mask] = targets[word_ids]

        return (inputs, expanded_targets)


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
    # TODO this needs to change
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