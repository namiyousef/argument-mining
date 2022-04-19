# -- public imports
import os

import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.testing import assert_frame_equal

import torch
from torch.utils.data import Dataset

import numpy as np

import warnings

# -- private imports
from argminer.utils import get_predStr

from mlutils.torchtools.helpers import unique_first


# -- datasets
class ArgumentMiningDataset(Dataset):
    """
    Class for loading data in batches and processing it
    """
    def __init__(self, df_label_map, df_text, tokenizer, max_length, strategy, is_train=True):
        super().__init__()


        # check data in correct format
        assert sorted(df_text.columns) == ['labels', 'text'], f"Please make sure input dataframe has the columns (text, labels)"

        # check strategy is within accepted strategies
        assert strategy in [
            'standard_io',
            'wordLevel_io', # consider deprecating wordLevel_io as it is the same as standard!
            'standard_bio',
            'wordLevel_bio',
            'standard_bixo',
            'standard_bieo',
            'wordLevel_bieo'
        ], 'Please use a valid strategy'

        # check that label map matches strategy
        assert 'O' in df_label_map.label.values, 'You are missing label "O"'

        strategy_level, strategy_name = strategy.split('_')

        labels = df_label_map.label[df_label_map.label.str.contains('-')].apply(lambda x: x[:1])
        unique_labels = sorted(list(labels.unique()))

        if strategy_name == 'io':
            assert 'I' == ''.join(unique_labels)
        elif strategy_name == 'bio':
            assert 'BI' == ''.join(unique_labels), 'You are missing one of labels "B" or "I"'
        elif strategy_name == 'bieo':
            assert 'BEI' == ''.join(unique_labels), 'You are missing one of labels "B", "I" or "E"'
        elif strategy_name == 'bixo':
            assert 'X' in df_label_map.label.values, 'You are missing label "X"'
            assert 'BI' == ''.join(unique_labels), 'You are missing one of labels "B" or "I"'
        else:
            raise NotImplementedError(f'Support for labelling strategy {strategy} does not exist yet')



        self.strategy_level = strategy_level
        self.strategy_name = strategy_name

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

        self.reduce_map = self.get_reduce_map()

        self.targets = df_text.labels.apply(
            lambda x: [self.label_to_id[label] for label in x]
        ).values

        # TODO padding not given label -100? Double check
        # TODO maybe also double check -1 ?


        self.is_train = is_train

        # -- prepare tokenizer
        self.tokenizer = tokenizer
        self.max_length = max_length


        # ignore index
        # TODO add option to add ignore_index to the subtokens as well? This will be a hassle tho
        self.ignore_index = -100

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
        word_ids_replaced = [word_id if word_id is not None else self.ignore_index for word_id in word_ids]

        targets = torch.as_tensor(targets, dtype=torch.long)

        labeller = getattr(self, f'_label_{self.strategy_level}') # TODO make function call directly

        targets = labeller(targets, word_id_mask, word_ids_filtered)
        targets = torch.as_tensor(targets, dtype=torch.long)

        if not self.is_train:
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
        expanded_targets = torch.ones(self.max_length, dtype=torch.long) * self.ignore_index
        expanded_targets[word_id_mask] = targets[word_ids]
        return expanded_targets

    # TODO can any of this be enhance using unique consequitive from torch?

    def _label_standard(self, targets, word_id_mask, word_ids):
        expanded_targets = self._label_wordLevel(targets, word_id_mask, word_ids)
        word_start_ids = unique_first(torch.as_tensor(word_ids))
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

    def get_reduce_map(self):
        reduce_map = {}
        label_to_id = {}
        for id_, label in self.id_to_label.items():
            if label not in ['X', 'O']:
                label = label.split('-')[1]
            reduce_map[id_] = label  # uses dict to preserve order
            label_to_id[label] = label
        label_to_id = {label: i for i, label in enumerate(label_to_id)}
        reduce_map = {i: label_to_id[label] for i, label in reduce_map.items()}
        return reduce_map


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


class BigBirdDataset(Dataset):
    """Dataset from the bigbird notebook (modified)
    """

    LABEL_ALL_SUBTOKENS = True
    output_labels = ['O', 'B-Lead', 'I-Lead', 'B-Position', 'I-Position', 'B-Claim', 'I-Claim', 'B-Counterclaim',
                     'I-Counterclaim',
                     'B-Rebuttal', 'I-Rebuttal', 'B-Evidence', 'I-Evidence', 'B-Concluding Statement',
                     'I-Concluding Statement']

    labels_to_ids = {v: k for k, v in enumerate(output_labels)}
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
                    label_ids.append(self.labels_to_ids[word_labels[word_idx]])
                else:
                    if self.LABEL_ALL_SUBTOKENS:
                        label_ids.append(self.labels_to_ids[word_labels[word_idx]])
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


# -- data processors
class DataProcessor:
    """
    Base class for getting data in a standardised form from raw
    """
    def __init__(self, path):
        self.path = path
        self.status = None
        self.dataframe = None

    def _process(self, strategy, processors=[]):

        df = self.dataframe.copy()

        for processor in processors:
            df['text'] = df['text'].apply(processor)

            # add predStr
        df = get_predStr(df)  # TODO double check start pred string here

        # add labelling strategy
        label_strat = dict(
            add_end='e' in strategy,
            add_beg='b' in strategy
        )
        df['label'] = df[['label', 'predictionString']].apply(
            lambda x: _generate_entity_labels(
                len(x['predictionString']), x['label'], **label_strat
            ), axis=1
        )

        self.dataframe = df
        self.status = 'processed'

        return self

    def _postprocess(self):

        df = self.dataframe.copy()

        df = df.groupby('doc_id').agg({
            'text': lambda x: ' '.join(x),
            'predictionString': 'sum',
            'label': 'sum'
        })

        df = df.reset_index().rename(columns={'label': 'labels'})[['text', 'labels']]

        self.dataframe = df
        self.status = 'postprocessed'

        return self



    @property
    def preprocess(self):
        """ This is pretokenisation cleaning / what is done at reading"""
        if self.status is not None:
            raise Exception('Preprocess method has already been called.')
        return self._preprocess

    @property
    def process(self):
        """ This is usually the tokeniser"""
        if self.status is None:
            raise Exception('Cannot run process before running preprocess')
        elif self.status != 'preprocessed':
            raise Exception('Process method has already been called.')

        return self._process

    @property
    def postprocess(self):
        """ This is post tokenisation cleaning"""
        if self.status is None:
            raise Exception('Cannot run postprocess before running process')
        elif self.status != 'processed':
            raise Exception('Postprocess method has already been called.')
        # TODO this is a BUG FIX
        return self._postprocess


    def from_json(self, status='postprocessed', df=None):
        assert status in {'preprocessed', 'processed', 'postprocessed'}

        if df is None:
            filename = f'{self.__class__.__name__.split("Processor")[0]}_{status}.json'
            path = os.path.join(self.path, filename)

            df = pd.read_json(path)

        self.dataframe = df
        self.status = status

        return self

    def save_json(self, dir_path=''):
        print("THIS BETTER SHOW")
        # TODO maybe add option for custom file name?
        filename = f'{self.__class__.__name__.split("Processor")[0]}_{self.status}.json'
        path = os.path.join(dir_path, filename)
        if self.dataframe is None:
            raise TypeError('Dataframe is not yet created yet. You must call preprocess')
        else:
            self.dataframe.to_json(path)

        return self


    def _default_tts(self, test_size, val_size=None, random_seed=0):
        # TODO need to add a fixed seed here...
        # TODO test and val sizes relative to initial dataset
        df = self.dataframe.copy()
        n_samples = df.doc_id.unique().shape[0]

        test_size = int(test_size * n_samples)
        if val_size:
            val_size = int(val_size * n_samples)
            assert test_size + val_size <= n_samples, 'total of test and val exceeds data size'

        dfs = {}
        np.random.seed(random_seed) 

        ids = np.arange(n_samples)
        np.random.shuffle(ids)

        test_ids = ids[:test_size]
        train_ids = ids[test_size:]

        doc_ids_test = df.doc_id.unique()[test_ids]
        doc_ids_train = df.doc_id.unique()[train_ids]

        df_test = df[df.doc_id.isin(doc_ids_test)]
        df_train = df[df.doc_id.isin(doc_ids_train)]


        # TODO val size based on old, broken
        '''dfs['test'] = df.loc[test_ids]

        if val_size:
            val_ids = train_ids[:val_size]
            train_ids = train_ids[val_size:]
            dfs['val'] = df.loc[val_ids]

        dfs['train'] = df.loc[train_ids]'''

        dfs['test'] = df_test
        dfs['train'] = df_train

        return dfs

    @property
    def get_tts(self):
        """ takes saved dataframe
        if dataframe in final state
        df_train, df_test (df_val option)
        """
        #if self.status != 'postprocessed':
        #    raise Exception('Cannot call train test split before postprocessing')

        if hasattr(self, '_get_tts'):#'_get_tts' in self.__dict__:
            # TODO do some of these need to be private?
            return self._get_tts
        else:
            return self._default_tts


class TUDarmstadtProcessor(DataProcessor):

    """
    Needs to have reading, e.g. doing stuff at reading stage

    Needs to have a step for doing things to the raw data, before labels are creating

    Needs to have a step for creating the labels

    Needs to have a step for making any changes after labels have been created
    """
    def __init__(self, path=''):
        super().__init__(path)


    def _preprocess(self):
        # TODO note for this that some of the files had to be modified, so might not work out of the box
        texts = []
        annotated_texts = []
        path = os.path.join(self.path, 'brat-project-final')
        for file in os.listdir(path):
            essay_num, file_extension = file.split('.')
            if file_extension == 'ann':
                with open(os.path.join(path, file), 'r') as f:
                    df_temp = pd.read_csv(f, delimiter='\t', header=None, names=['label_type', 'label', 'text'])
                    df_temp[['label', 'label_comp1', 'label_comp2']] = df_temp.label.str.split(expand=True)
                    df_temp['doc_id'] = essay_num
                    annotated_texts.append(df_temp)
            elif file_extension == 'txt':
                with open(os.path.join(path, file), 'r') as f:
                    texts.append((essay_num, f.read()))
            else:
                continue

        df_texts = pd.DataFrame.from_records(texts, columns=['doc_id', 'text'])
        df_annotated = pd.concat(annotated_texts)

        assert sorted(df_annotated.doc_id.unique()) == sorted(df_texts.doc_id)

        ids_argument_segment = df_annotated.label_type.str.startswith('T')
        df_arguments = df_annotated[ids_argument_segment]
        df_arguments = df_arguments.rename(columns={'label_comp1': 'span_start', 'label_comp2': 'span_end'}).astype(
            {'span_start': int, 'span_end': int}
        )
        records = []
        df_arguments = df_arguments.sort_values(['doc_id', 'span_start', 'span_end'])
        for (doc_id, text) in df_texts.sort_values('doc_id').itertuples(index=False):
            df_argument = df_arguments[df_arguments.doc_id == doc_id]
            prev_span = 0
            for i, (text_segment, span_start, span_end) in enumerate(
                    df_argument[['text', 'span_start', 'span_end']].itertuples(index=False)):
                try:
                    assert text_segment == text[span_start: span_end]
                except Exception as e:
                    print(f'Found non-matching segments:{"-"*50}\n\n'
                          f'{text_segment}\n\n'
                          f'{text[span_start: span_end]}\n')

                    df_arguments['text'] = df_arguments['text'].where(df_arguments['text'] != text_segment, text[span_start: span_end])
                    # all the exception were manually checked. These are because of qutoe chars, this is a hot fix.!!!! TODO

                records.append(('O', 'Other', text[prev_span: span_start], prev_span, span_start, doc_id))

                prev_span = span_end
            records.append(('O', 'Other', text[prev_span:], prev_span, len(text), doc_id))

        df_other = pd.DataFrame.from_records(records, columns=df_arguments.columns)

        df_combined = pd.concat([df_other, df_arguments])
        df_combined = df_combined.sort_values(['doc_id', 'span_start', 'span_end']).reset_index(drop=True)[['doc_id', 'text', 'label']]

        # TODO move test maybe?
        assert_frame_equal(
            df_combined.groupby('doc_id').agg({'text':lambda x: ''.join(x)}).reset_index(),
            df_texts.sort_values('doc_id').reset_index(drop=True)
        )

        self.dataframe = df_combined
        self.status = 'preprocessed'

        return self

    # make it possible to run process and postprocess after preprocess as been run!

    def _process(self, strategy, processors=[], split='all', **split_params):
        # processes data to standardised format, adds any extra cleaning steps
        assert strategy in {'io', 'bio', 'bieo'} # for now
        assert split in {'all', 'train', 'test'}

        if split == 'all':
            df = self.dataframe.copy()
        else:
            df_dict = self.get_tts(**split_params)
            df = df_dict[split]
            warnings.warn(f'Getting data for split={split} with params {split_params}', UserWarning, stacklevel=2)

        for processor in processors:
            df['text'] = df['text'].apply(processor)

        # add predStr
        df = get_predStr(df) # TODO double check start pred string here

        # add labelling strategy
        label_strat = dict(
            add_end='e' in strategy,
            add_beg='b' in strategy
        )
        df['label'] = df[['label', 'predictionString']].apply(
            lambda x: _generate_entity_labels(
                len(x['predictionString']), x['label'], **label_strat
            ), axis=1
        )

        self.dataframe = df
        self.status = 'processed'


        return self


    def _postprocess(self):
        # aggregates data

        df = self.dataframe.copy()
        

        df = df.groupby('doc_id').agg({
            'text':lambda x: ' '.join(x),
            'predictionString': 'sum',
            'label': 'sum'
        })

        df = df.reset_index().rename(columns={'label':'labels'})

        self.dataframe = df
        self.status = 'postprocessed'

        return self

    def _get_tts(self, val_size=None, **kwargs):
        # TODO this will cause a bug when running if you do load from json
        # because it will NOT be able to see the train_test_split ids...
        # This needs reworking / thinking!
        # TODO kwargs added to avoid problem when running all processors together
        """
        Train test split based on the TUDarmstadt dataset tts file
        :return:
        """
        path = os.path.join(self.path, 'train-test-split.csv')
        df_tts_ids = pd.read_csv(path, delimiter=';')
        ids_train = df_tts_ids.SET == "TRAIN"
        ids_test = ~ids_train

        dfs = {}
        df = self.dataframe.copy()
        if self.status == 'postprocessed':
            df_train = df[ids_train]
            df_test = df[ids_test]
            dfs['test'] = df_test
            dfs['train'] = df_train
        elif self.status == 'preprocessed':
            doc_ids_test = df_tts_ids.ID.values[ids_test]
            df_test = df[df.doc_id.isin(doc_ids_test)]
            df_train = df[~df.doc_id.isin(doc_ids_test)]
            dfs['test'] = df_test
            dfs['train'] = df_train


        # TODO this is now broken because of hotfix above, need unittests
        n_samples = df.shape[0]
        if val_size:
            val_size = int(val_size * n_samples)
            n_test_samples = df_test.shape[0]
            assert val_size + n_test_samples <= n_samples, 'selected val_size is greater than the training set. ' \
                                                          f'The test set covers {n_test_samples}/{n_samples} of data'
            ids_train = df_train.index.values
            ids_val = ids_train[:val_size]
            ids_train = ids_train[val_size:]
            dfs['val'] = df_train.loc[ids_val]
            dfs['train'] = df_train.loc[ids_train]

        return dfs
    

class PersuadeProcessor(DataProcessor):
    def __init__(self,path=''):
        super().__init__(path)
        
    def _preprocess(self):
        path_to_text_dir = os.path.join(self.path, 'train')
        path_to_ground_truth = os.path.join(self.path, 'train.csv')
        
        df_ground_truth = pd.read_csv(path_to_ground_truth)
        df_texts = df_from_text_files(path_to_text_dir)
        
        df_ground_truth = df_ground_truth.sort_values(['id', 'discourse_start', 'discourse_end'])
        df_ground_truth = df_ground_truth.drop(columns=['discourse_id','discourse_start','discourse_end','discourse_type_num'])
        df_ground_truth = df_ground_truth.rename(columns={'discourse_type':'label','predictionstring':'predictionString',
                                                  'discourse_text':'text','id':'doc_id'})

        df_texts = df_texts.rename(columns={'text':'doc_text','id':'doc_id'})
        df_texts['text_split'] = df_texts.doc_text.str.split()
        df_texts['range'] = df_texts['text_split'].apply(lambda x: list(range(len(x))))
        df_texts['start_id'] = df_texts['range'].apply(lambda x: x[0])
        df_texts['end_id'] = df_texts['range'].apply(lambda x: x[-1])
        df_texts = df_texts.drop(columns=['text_split','range'])
        
        df_ground_truth['predictionString'] = df_ground_truth.predictionString.apply(lambda x: [int(num) for num in x.split()])
        df_ground_truth['pred_str_start_id'] = df_ground_truth.predictionString.apply(lambda x: x[0])
        df_ground_truth['pred_str_end_id'] =  df_ground_truth.predictionString.apply(lambda x: x[-1])
        
        df = df_ground_truth.merge(df_texts)
        
        
        new = []
        df = df.sort_values(['doc_id','pred_str_start_id'])
        prev_doc = df.iloc[0].doc_id
        gd_end = -1
        
        for row in df.itertuples(index=False):

            if row.doc_id != prev_doc:

                prev_row = df[(df.doc_id == prev_doc) & (df.pred_str_end_id == gd_end)].squeeze()

                if prev_row.pred_str_end_id != prev_row.end_id:

                    s = prev_row.doc_text.split()[prev_row.pred_str_end_id+1:]
                    new_string = ' '.join(s)
                    new_predsStr = list(range(prev_row.pred_str_end_id+1,prev_row.end_id+1))
                    new_type = 'Other'
                    new_id = prev_row.doc_id
                    new_row = {'doc_id': new_id, 'text':new_string ,'label':'Other',
                               'predictionString':new_predsStr,'pred_str_start_id':new_predsStr[0],
                              'pred_str_end_id':new_predsStr[-1],'doc_text':prev_row.doc_text,
                              'start_id':prev_row.start_id,'end_id':prev_row.end_id}
                    new.append(new_row)


                if row.pred_str_start_id != row.start_id:
                    s = row.doc_text.split()[:row.pred_str_start_id]
                    new_string = ' '.join(s)
                    new_predsStr = list(range(row.start_id,row.pred_str_start_id))
                    new_type = 'Other'
                    new_id = row.doc_id
                    new_row = {'doc_id': new_id, 'text':new_string ,'label':'Other',
                               'predictionString':new_predsStr,'pred_str_start_id':row.start_id,
                              'pred_str_end_id':row.pred_str_start_id-1,'doc_text':row.doc_text,
                              'start_id':row.start_id,'end_id':row.end_id}
                    new.append(new_row)

                prev_doc = row.doc_id
                gd_end = row.pred_str_end_id 

            else: #stay in the same doc
                if row.pred_str_start_id != (gd_end+1) :
                    s = row.doc_text.split()[gd_end+1:row.pred_str_start_id]
                    new_string = ' '.join(s)
                    new_predsStr = list(range(gd_end+1,row.pred_str_start_id))
                    new_type = 'Other'
                    new_id = row.doc_id
                    new_row = {'doc_id': new_id, 'text':new_string ,'label':'Other',
                               'predictionString':new_predsStr,'pred_str_start_id':gd_end+1,
                              'pred_str_end_id':row.pred_str_start_id-1,'doc_text':row.doc_text,
                              'start_id':row.start_id,'end_id':row.end_id}
                    new.append(new_row)


                gd_end = row.pred_str_end_id  #line 7


        df = pd.concat([df,pd.DataFrame().from_records(new)])
        df = df.sort_values(['doc_id','pred_str_start_id','pred_str_end_id'])
        df = df.drop(columns=['doc_text','start_id','end_id','pred_str_start_id','pred_str_end_id'])
        df = df.reset_index(drop=True)

        self.dataframe = df
        self.status = 'preprocessed'
        return self
    
    def _process(self, strategy, processors=[], split='all', **split_params):
        assert split in {'all', 'train', 'test'}
        # processes data to standardised format, adds any extra cleaning steps
        assert strategy in {'io', 'bio', 'bieo'} # for now

        # TODO imrpove this
        if split == 'all':
            df = self.dataframe.copy()
        else:
            df_dict = self.get_tts(**split_params)
            df = df_dict[split]
            print('After split: ', df.shape)
            warnings.warn(f'Getting data for split={split} with params {split_params}', UserWarning, stacklevel=2)



        for processor in processors:
            df['text'] = df['text'].apply(processor)


        # add labelling strategy
        label_strat = dict(
            add_end='e' in strategy,
            add_beg='b' in strategy
        )
        
        df = get_predStr(df) 
        
        df['label'] = df[['label', 'predictionString']].apply(
            lambda x: _generate_entity_labels(
                len(x['predictionString']), x['label'], **label_strat
            ), axis=1
        )
        print('before save in process: ', df.shape)

        self.dataframe = df
        print('after save in process: ', df.shape)

        self.status = 'processed'


        return self
    
    def _postprocess(self):
        print('before load in postprocess: ', self.dataframe.shape)

        df_post = self.dataframe.copy()
        print('after load in postprocess: ', df_post.shape)

        df_post = df_post.groupby('doc_id').agg({
            'text':lambda x: ' '.join(x),
            'predictionString': 'sum',
            'label': 'sum'
        }).reset_index()
        
        df_post = df_post.rename(columns={'label':'labels'})

        print('before save in postprocess: ', df_post.shape)

        self.dataframe = df_post
        print('after save in postprocess: ', self.dataframe.shape)

        self.status = 'postprocessed'

        return self
# class PersuadeProcessor(DataProcessor):

#     def __init__(self, path=''):
#         super().__init__(path)

#     def _preprocess(self):
#         warnings.warn('PersuadeProcessor does not have a preprocessor. '
#                       'Instead the postprocess method will prepare the data end-to-end', stacklevel=2)

#         self.status = 'preprocessed'
#         return self

#     def _process(self, strategy, processors=[]):
#         warnings.warn('PersuadeProcessor does not have a processor. '
#                       'Instead the postprocess method will prepare the data end-to-end', stacklevel=2)
#         if processors:
#             # TODO need to change how processor work. This is a hotfix because doing this parsing correctly
#             # is difficult due to corrupted discourse_start and end values. See https://www.kaggle.com/competitions/feedback-prize-2021/discussion/297688
#             warnings.warn('PersuadeProcessor does NOT accept any processors at this time.', stacklevel=2)

#         # TODO this is for postprocess
#         assert strategy in {'io', 'bio', 'bieo'} # for now

#         # add labelling strategy
#         label_strat = dict(
#             add_end='e' in strategy,
#             add_beg='b' in strategy
#         )
#         self.label_strat = label_strat

#         self.status = 'processed'
#         return self

#     def _postprocess(self):
#         warnings.warn('The postprocess method is behaving in a special way because of data corruption. '
#                       'This behaviour will change in the future.', DeprecationWarning, stacklevel=2)

#         label_strat = self.label_strat
#         path_to_text_dir = os.path.join(self.path, 'train')
#         path_to_ground_truth = os.path.join(self.path, 'train.csv')

#         df = create_labels_doc_level(path_to_text_dir, path_to_ground_truth, **label_strat)

#         df = df[['id', 'text', 'labels']].rename(columns={'id':'doc_id'})

#         self.dataframe = df
#         self.status = 'postprocessed'
#         return self

# -- helpers (may move later)
def _generate_entity_labels(length, label, add_end=False, add_beg=True):
    # TODO can this get length=0 as input? (this is a pipeline / integration test)
    """
    For cases where argument segment is only 1 word long, beginning given preference over end
    """
    labels = [f'I-{label}'] if label != 'Other' else ['O']
    labels *= length

    if add_end:
        if label != 'Other':
            labels[-1] = f'E-{label}'

    if add_beg:
        if label != 'Other':
            labels[0] = f'B-{label}'

    return labels


def df_from_text_files(path_to_dir):
    filenames = [filename for filename in os.listdir(path_to_dir)]
    records = [(filename.rstrip('.txt'), open(os.path.join(path_to_dir, filename), 'r').read()) for filename in filenames]
    df = pd.DataFrame.from_records(records, columns=['id', 'text'])
    return df

def create_labels_doc_level(
        path_to_text_dir,
        path_to_ground_truth,
        add_end=False,
        add_beg=True
):

    df_ground_truth = pd.read_csv(path_to_ground_truth)


    df_ground_truth.predictionstring = df_ground_truth.predictionstring.str.split()
    df_ground_truth['label_ids'] = df_ground_truth.predictionstring.apply(lambda x: [int(x[0]), int(x[-1])])
    df_ground_truth['range'] = df_ground_truth.label_ids.apply(lambda x: np.arange(x[0], x[1]+1))


    df_ground_truth['labels'] = df_ground_truth[['discourse_type', 'range']].apply(
        lambda x: _generate_entity_labels(len(x.range), x.discourse_type, add_end, add_beg),
        axis=1
    )

    df_texts = df_from_text_files(path_to_text_dir)
    df_texts.text = df_texts.text.str.strip()
    df_texts['text_split'] = df_texts.text.str.split()
    df_texts['labels'] = df_texts.text_split.apply(lambda x: len(x)*['O'])
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

    return df_texts

if __name__ == '__main__':

    processor = DataProcessor('test').preprocess()

    #processor = processor.preprocess().process('bio', split='test').postprocess()
    #df_test = processor.dataframe

    #processor = TUDarmstadtProcessor('../data/UCL/dataset2/ArgumentAnnotatedEssays-2.0')
    #processor = processor.preprocess().process('bio').postprocess()
    #df_test_new = processor.get_tts()['test']

    #from pandas.testing import assert_frame_equal
    #assert_frame_equal(df_test_new.reset_index(drop=True),
    #                   df_test.reset_index(drop=True))


