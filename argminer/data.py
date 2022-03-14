from torch.utils.data import Dataset
import torch
from pandas.api.types import is_string_dtype

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

        inputs = {
            key: torch.as_tensor(val, dtype=torch.long) for key, val in inputs.items()
        }
        targets = torch.as_tensor(targets, dtype=torch.long)
        expanded_targets = torch.zeros(self.max_length, dtype=torch.long)
        expanded_targets[word_id_mask] = targets[word_ids]

        return (inputs, expanded_targets)


LABEL_ALL_SUBTOKENS = True
output_labels = ['O', 'B-Lead', 'I-Lead', 'B-Position', 'I-Position', 'B-Claim', 'I-Claim', 'B-Counterclaim',
                 'I-Counterclaim',
                 'B-Rebuttal', 'I-Rebuttal', 'B-Evidence', 'I-Evidence', 'B-Concluding Statement',
                 'I-Concluding Statement']

labels_to_ids = {v: k for k, v in enumerate(output_labels)}

class BigBirdDataset(Dataset):
    """Dataset from the bigbird notebook (modified)
    """


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
        item = {key: torch.as_tensor(val, dtype=torch.long) for key, val in encoding.items()}
        if self.get_wids:
            word_ids2 = [w if w is not None else -1 for w in word_ids]
            item['wids'] = torch.as_tensor(word_ids2)

            return item
        else:
            targets = encoding.pop('labels')
            inputs = encoding
            return inputs, targets

    def __len__(self):
        return self.len