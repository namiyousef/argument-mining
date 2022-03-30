from torch.utils.data import Dataset
import torch
from pandas.api.types import is_string_dtype

from argminer.utils import _first_appearance_of_unique_item

class ArgumentMiningDataset(Dataset):
    """
    Class for loading data in batches and processing it
    """
    def __init__(self, df_label_map, df_text, tokenizer, max_length, strategy, is_train=True):
        super().__init__()

        assert sorted(df_text.columns) == ['labels', 'text'], f"Please make sure input dataframe has the columns (text, labels)"
        assert 'O' in df_label_map.label.values, 'You are missing label "O"'
        assert strategy in ['standard_bio', 'wordLevel_bio', 'standard_bixo', 'standard_bieo', 'wordLevel_bieo'], 'Please use a valid strategy'

        strategy_level, strategy_name = strategy.split('_') # TODO this will cause a bug

        self.strategy_level = strategy_level
        self.strategy_name = strategy_name

        if strategy_name == 'bio':
            labels = df_label_map.label[df_label_map.label.str.contains('-')].apply(lambda x: x[:1])
            labels = list(labels.unique())
            assert 'BI' == ''.join(sorted(labels)), 'You are missing one of labels "B" or "I"'
        elif strategy_name == 'bieo':
            labels = df_label_map.label[df_label_map.label.str.contains('-')].apply(lambda x: x[:1])
            labels = list(labels.unique())
            assert 'BIE' == ''.join(labels), 'You are missing one of labels "B", "I" or "E"'
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

        # create hash tables
        self.label_to_id = {
            label: id_ for label, id_ in df_label_map[['label', 'label_id']].values
        }
        self.id_to_label = {
            id_: label for label, id_ in self.label_to_id.items()
        }


        self.targets = df_text.labels.apply(
            lambda x: [self.label_to_id[label] for label in x]
        ).values


        self.is_train = is_train

        # -- prepare tokenizer
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.inputs)
    # TODO need to add option to separate how CLS, PAD and SEP are labelled
    # TODO need to add option for ignoring BIXO using attention mask
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

        word_ids_filtered = [word_id for word_id in word_ids if word_id is not None]
        word_ids_replaced = [word_id if word_id is not None else -1 for word_id in word_ids]

        targets = torch.as_tensor(targets, dtype=torch.long)

        labeller = getattr(self, f'_label_{self.strategy_level}') # TODO make function call directly

        targets = labeller(targets, word_id_mask, word_ids_filtered)
        targets = torch.as_tensor(targets, dtype=torch.long)

        inputs['word_ids'] = word_ids_replaced
        inputs['index'] = torch.as_tensor(index)

        # for training, no need to return word_ids, or word_id_mask
        # for validation and testing, there is a need to return them!

        # TODO need to think about reading weights in the middle while training?
        # TODO need to think about sending things to the GPU, which ones to send
        inputs = {
            key: torch.as_tensor(val, dtype=torch.long) for key, val in inputs.items()
        }

        #inputs['doc_ids'] = index  # TODO may have to convert to type long
        #inputs['word_ids'] = word_ids



        return (inputs, targets)

    def _label_wordLevel(self, targets, word_id_mask, word_ids):
        expanded_targets = torch.zeros(self.max_length, dtype=torch.long)
        expanded_targets[word_id_mask] = targets[word_ids]
        return expanded_targets

    # TODO can any of this be enhance using unique consequitive from torch?

    def _label_standard(self, targets, word_id_mask, word_ids):
        expanded_targets = torch.zeros(self.max_length, dtype=torch.long)
        # TODO call _label_wordLevel here

        expanded_targets[word_id_mask] = targets[word_ids]
        word_start_ids = _first_appearance_of_unique_item(torch.as_tensor(word_ids))
        unique_word_ids, word_id_counts = torch.unique(torch.as_tensor(word_ids), return_counts=True)

        expanded_targets_with_mask = expanded_targets[word_id_mask]
        for i, (word_start_id, word_id, word_id_count) in enumerate(
                zip(word_start_ids, unique_word_ids, word_id_counts)):
            curr_target = expanded_targets_with_mask[word_start_id].item()
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
                    expanded_targets_with_mask[ids] = expanded_target_label # this label needs to be changed!

        expanded_targets[word_id_mask] = expanded_targets_with_mask
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
        word_id_mask = [word_id is not None for word_id in word_ids] # consider switching mask
        # to the indices that need to be read
        word_ids_filtered = [word_id for word_id in word_ids if word_id is not None]

        inputs['word_ids'] = [word_id if word_id is not None else -1 for word_id in word_ids]


        inputs = {
            key: torch.as_tensor(val, dtype=torch.long) for key, val in inputs.items()
        }
        # TODO you don't convert these to tensors!
        inputs['word_id_mask'] = word_id_mask # TODO not agged properly

        targets = torch.as_tensor(targets, dtype=torch.long)
        expanded_targets = torch.zeros(self.max_length, dtype=torch.long)
        expanded_targets[word_id_mask] = targets[word_ids_filtered]

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
            targets = item.pop('labels')
            inputs = item
            return inputs, targets

    def __len__(self):
        return self.len