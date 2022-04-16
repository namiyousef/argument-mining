import unittest
from argminer.data import ArgumentMiningDataset
from argminer.utils import _get_label_maps
from transformers import AutoTokenizer
import pandas as pd
import torch

from itertools import product


test_string = 'I am Yousef Nami. NLP is the fastest growing Machine Learning field. NLP to science direct is it the most fire field in Machine NLP. Some other sentences. NLP, I conclude that NLP is NLP'
labels = ['Other', 'Claim', 'Evidence', 'Other', 'Conclusion']
def create_labelled_stentences(string, labels, strategy):
    sentences = string.split('.')
    output_labels = []
    for sentence, label in zip(sentences, labels):
        print(sentence, label)
        if label == 'Other':
            sentence_labels = len(sentence.split()) * ['O']
        else:
            sentence_labels = len(sentence.split()) * [f'I-{label}']
            if strategy == 'bieo':
                sentence_labels[-1] = f'E-{label}'
            sentence_labels[0] = f'B-{label}'
        output_labels += sentence_labels
    return pd.DataFrame({'text':[' '.join(string)], 'labels':[output_labels]})


class TestDataset(unittest.TestCase):

    def test_dataset_bigbird_tokenizer(self):
        model_name = 'google/bigbird-roberta-large'
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # tests
        strategy_names = ['bio', 'bieo', 'bixo']
        strategy_levels = ['wordLevel', 'standard']


        strategies = [
            f'{strategy_level}_{strategy_name}' for strategy_name in strategy_names for strategy_level in strategy_levels if f'{strategy_level}_{strategy_name}' != 'word_level_bixo'
        ]
        label_maps = {
            strategy: _get_label_maps(labels, strategy.split('_')[-1]) for strategy in strategies
    }

        test_string = 'I am Yousef Nami ' \
                      'NLP is the fastest growing Machine Learning field ' \
                      'NLP to science direct is it the most fire field in Machine NLP ' \
                      'Some other sentences ' \
                      'NLP I conclude that NLP is NLP'

        bio_labels = ['O']*4 +['B-Claim']+['I-Claim']*7 + ['B-Evidence']+['I-Evidence']*12 + ['O']*3 + ['B-Conclusion']+['I-Conclusion']*6
        bieo_labels = ['O']*4 +['B-Claim']+['I-Claim']*6+['E-Claim'] + ['B-Evidence']+['I-Evidence']*11+['E-Evidence'] + ['O']*3 + ['B-Conclusion']+['I-Conclusion']*5+['E-Conclusion']

        text_labels = dict(
            wordLevel_bio=bio_labels,
            standard_bio=bio_labels,
            wordLevel_bieo=bieo_labels,
            standard_bieo=bieo_labels,
            standard_bixo=bio_labels
        )

        text_dfs = {
            key: pd.DataFrame({
                "labels": [text_labels[key]],
                "text":test_string}) for key in text_labels
        }

        max_length = len(tokenizer(test_string).word_ids())
        print(tokenizer.tokenize(test_string))
        expected_output = dict(
            wordLevel_bio=[0]*7 + [1]*2 + [2]*7 + [3]*2 + [4]*13 + [0]*3 + [5]*2 + [6]*8,
            wordLevel_bieo=[0]*7 + [1]*2 + [2]*6 + [3]*1 + [4]*2 + [5]*11 + [6]*2 + [0]*3 + [7]*2 + [8]*6 + [9]*2,
            standard_bio=[0]*7 + [1] + [2]*8 + [3]*1 + [4]*14 + [0]*3 + [5]*1 + [6]*9,
            standard_bieo=[0]*7 + [1] + [2]*7 + [3]*1 + [4]*1 + [5]*13 + [6]*1 + [0]*3 + [7]*1 + [8]*8 + [9]*1,
            standard_bixo=[0]*7 + [2] + [1] + [3]*7 + [4] + [1] + [5]*12 + [1] + [0]*3 + [6] + [1] + [7]*4 + [1] + [7]*2+ [1]
        )
        for i, strategy in enumerate(strategies):
            if strategy in expected_output:
                dataset = ArgumentMiningDataset(label_maps[strategy], text_dfs[strategy], tokenizer, max_length, strategy)
                for (inputs, targets) in dataset:
                    targets = targets[targets != -100]
                    expected_targets = torch.as_tensor(expected_output[strategy])
                    assert (targets == expected_targets).all(), f'assertion failed for strategy: {strategy}:\n{targets}\n{expected_targets}'

    def test_argument_mining_dataset(self):

        # -- configuration

        class TokenizerMock:
            def __init__(self):
                # set random how to tokenize

                class Inputs:
                    def __init__(self, input_ids, attention_mask, word_id_list):
                        self.data = dict(
                            input_ids=input_ids,
                            attention_mask=attention_mask
                        )
                        self.word_id_list = word_id_list

                    def __getitem__(self, key):
                        return self.data[key]

                    def __setitem__(self, key, value):
                        self.data[key] = value

                    def word_ids(self):
                        return self.word_id_list

                    def items(self):
                        return [(key, val) for key, val in self.data.items()]

                self.inputs = Inputs
                self.vocab = {'PAD': 0, 'CLS': 1, 'ing': 2, 'er': 3, 'es': 4, 'SEP':5, 'UNK': 6}


            def __call__(self, word_split, max_length, **kwargs):
                input_ids = [self.vocab['CLS']]
                attention_mask = [1]
                word_id_list = [None]
                if max_length == 2:
                    return self.inputs(input_ids+[self.vocab['SEP']], attention_mask+[1], word_id_list+[None])
                else:
                    for i, word in enumerate(word_split):
                        for j, token in enumerate(self.vocab):
                            if token in word:
                                input_ids.append(self.vocab['UNK'])
                                word_id_list.append(i)
                                attention_mask.append(1)

                                input_ids.append(self.vocab[token])
                                word_id_list.append(i)
                                attention_mask.append(1)
                                break
                            elif j == len(self.vocab) - 1:
                                if word not in self.vocab:
                                    input_ids.append(self.vocab['UNK'])
                                word_id_list.append(i)
                                attention_mask.append(1)
                    if max_length == 1:
                        input_ids.append(self.vocab['CLS'])
                        attention_mask.append(i)
                        word_id_list.append(None)
                    else:
                        if max_length < len(input_ids):
                            # cut
                            input_ids = input_ids[:max_length]
                            attention_mask = attention_mask[:max_length]
                            word_id_list = word_id_list[:max_length]
                            input_ids[-1] = self.vocab['SEP']
                            word_id_list[-1] = None
                        else:
                            if max_length == len(input_ids):
                                input_ids[-1] = self.vocab['SEP']
                                word_id_list[-1] = None
                            else:
                                input_ids.append(self.vocab['SEP'])
                                word_id_list.append(None)
                                attention_mask.append(1)
                            for i in range(max_length - len(input_ids)):
                                input_ids.append(self.vocab['PAD'])
                                attention_mask.append(0)
                                word_id_list.append(None)
                        return self.inputs(input_ids, attention_mask, word_id_list)

        tokenizer = TokenizerMock()
        print(tokenizer('Rares Mohammad Nami. I bring her with Rares'.split(), 15).items())
        # todo tokenizer needs testing

        df_text = pd.DataFrame(dict(
            text=['Rares Mohammad Rares I bring her with Rares']
        ))

        df_text_dict = dict(
            io=df_text.assign(
                labels=[['I-PERS', 'I-PERS', 'I-PERS', 'O', 'O', 'O', 'O', 'I-PERS']]
            ),
            bio=df_text.assign(
                labels=[['B-PERS', 'I-PERS', 'I-PERS', 'O', 'O', 'O', 'O', 'B-PERS']]
            ),
            bieo=df_text.assign(
                labels=[['B-PERS', 'I-PERS', 'E-PERS', 'O', 'O', 'O', 'O', 'B-PERS']]
            ),
            bixo=df_text.assign(
                labels=[['B-PERS', 'I-PERS', 'I-PERS', 'O', 'O', 'O', 'O', 'B-PERS']]
            )
        )

        df_label_map_dict = dict(
            io=pd.DataFrame(dict(
                label_id=[0, 1],
                label=['O', 'I-PERS']
            )),
            bio=pd.DataFrame(dict(
                label_id=[0, 1, 2],
                label=['O', 'B-PERS', 'I-PERS']
            )),
            bieo=pd.DataFrame(dict(
                label_id=[0, 1, 2, 3],
                label=['O','B-PERS', 'I-PERS', 'E-PERS']
            )),
            bixo=pd.DataFrame(dict(
                label_id=[0, 1, 2, 3],
                label=['O','X', 'B-PERS', 'I-PERS']
            ))
        )

        max_length_dict=dict(
            MaxLengthLess=6,
            MaxLengthEqual=15,
            MaxLengthMore=17,
        )

        # Actual: Rares Mohammad Rar es I bring her with Rares'
        # Tokenised:  Ra res Mohammad Rar es I br ing h er with Ra res'
        # word_ids:   0   0         1   2  2 3  4   4 5  5    6  7   7
        # input_ids:  6   4         6   6  4 6  6   2 6  3    6  6   4
        # targets:    ?   ?         ?   ?  ? 0  0   0 0  0    0  ?   ?
        # length less = 6, equal = 13, more =15

        inputs_dict = dict(
            MaxLengthLess=dict(
                    input_ids=torch.tensor([
                        1,
                        6, 4, 6, 6,
                        5
                    ]),
                    word_ids=torch.tensor([
                        -100,
                        0, 0, 1, 2,
                        -100
                    ]),
                    attention_mask=torch.tensor([
                        1,
                        1, 1, 1, 1,
                        1
                    ]),
                    index=torch.tensor([0])
            ),
            MaxLengthEqual=dict(
                    input_ids=torch.tensor([
                        1,
                        6, 4, 6, 6, 4,
                        6, 6, 2, 6, 3, 6,
                        6, 4,
                        5
                    ]),
                    word_ids=torch.tensor([
                        -100,
                        0, 0, 1, 2, 2,
                        3, 4, 4, 5, 5, 6,
                        7, 7,
                        -100
                    ]),
                    attention_mask=torch.tensor([
                        1,
                        1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1,
                        1, 1,
                        1
                    ]),
                    index=torch.tensor([0])
            ),
            MaxLengthMore=dict(
                    input_ids=torch.tensor([
                        1,
                        6, 4, 6, 6, 4,
                        6, 6, 2, 6, 3, 6,
                        6, 4,
                        5,
                        0, 0
                    ]),
                    word_ids=torch.tensor([
                        -100,
                        0, 0, 1, 2, 2,
                        3, 4, 4, 5, 5, 6,
                        7, 7,
                        -100,
                        -100, -100
                    ]),
                    attention_mask=torch.tensor([
                        1,
                        1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1,
                        1, 1,
                        1,
                        0, 0
                    ])
            )
        )
        expected_outputs = dict(
            standard_io_MaxLengthLess=dict(
                inputs=inputs_dict['MaxLengthLess'],
                targets=torch.tensor([
                    -100,
                    1, 1, 1, 1,
                    -100
                ])
            ),
            standard_io_MaxLengthEqual=dict(
                inputs=inputs_dict['MaxLengthEqual'],
                targets=torch.tensor([
                    -100,
                    1, 1, 1, 1, 1,
                    0, 0, 0, 0, 0, 0,
                    1, 1,
                    -100
                ])
            ),
            standard_io_MaxLengthMore=dict(
                inputs=inputs_dict['MaxLengthMore'],
                targets=torch.tensor([
                    -100,
                    1, 1, 1, 1, 1,
                    0, 0, 0, 0, 0, 0,
                    1, 1,
                    -100,
                    -100, -100
                ])
            ),
            wordLevel_io_MaxLengthLess=dict(
                inputs=inputs_dict['MaxLengthLess'],
                targets=torch.tensor([
                    -100,
                    1, 1, 1, 1,
                    -100
                ])
            ),
            wordLevel_io_MaxLengthEqual=dict(
                inputs=inputs_dict['MaxLengthEqual'],
                targets=torch.tensor([
                    -100,
                    1, 1, 1, 1, 1,
                    0, 0, 0, 0, 0, 0,
                    1, 1,
                    -100
                ])
            ),
            wordLevel_io_MaxLengthMore=dict(
                inputs=inputs_dict['MaxLengthMore'],
                targets=torch.tensor([
                    -100,
                    1, 1, 1, 1, 1,
                    0, 0, 0, 0, 0, 0,
                    1, 1,
                    -100,
                    -100, -100
                ])
            ),
            standard_bio_MaxLengthLess=dict(
                inputs=inputs_dict['MaxLengthLess'],
                targets=torch.tensor([
                    -100,
                    1, 2, 2, 2,
                    -100
                ])
            ),
            # Actual: Rares Mohammad Nami. I bring her with Rares'
            # Tokenised:  Ra res Mohammad Nami. I br ing h er with Ra res'
            # word_ids:   0   0        1    1  2  3   3 4  4    5  6   6
            # input_ids:  6   4        6    6  6  6   2 6  3    6  6   4
            # targets:    ?   ?        ?    ?  0  0   0 0  0    0  ?   ?
            # length less = 6, equal = 13, more =15
            standard_bio_MaxLengthEqual=dict(
                inputs=inputs_dict['MaxLengthEqual'],
                targets=torch.tensor([
                    -100,
                    1, 2, 2, 2, 2,
                    0, 0, 0, 0, 0, 0,
                    1, 2,
                    -100
                ])
            ),
            standard_bio_MaxLengthMore=dict(
                inputs=inputs_dict['MaxLengthMore'],
                targets=torch.tensor([
                    -100,
                    1, 2, 2, 2, 2,
                    0, 0, 0, 0, 0, 0,
                    1, 2,
                    -100,
                    -100, -100
                ])
            ),
            wordLevel_bio_MaxLengthLess=dict(
                inputs=inputs_dict['MaxLengthLess'],
                targets=torch.tensor([
                    -100,
                    1, 1, 2, 2,
                    -100
                ])
            ),
            wordLevel_bio_MaxLengthEqual=dict(
                inputs=inputs_dict['MaxLengthEqual'],
                targets=torch.tensor([
                    -100,
                    1, 1, 2, 2, 2,
                    0, 0, 0, 0, 0, 0,
                    1, 1,
                    -100
                ])
            ),
            wordLevel_bio_MaxLengthMore=dict(
                inputs=inputs_dict['MaxLengthMore'],
                targets=torch.tensor([
                    -100,
                    1, 1, 2, 2, 2,
                    0, 0, 0, 0, 0, 0,
                    1, 1,
                    -100,
                    -100, -100
                ])
            ),

            standard_bieo_MaxLengthLess=dict(
                inputs=inputs_dict['MaxLengthLess'],
                targets=torch.tensor([
                    -100,
                    1, 2, 2, 3, # TODO double check if this is the behaviour you want!
                    -100
                ])
            ),
            standard_bieo_MaxLengthEqual=dict(
                inputs=inputs_dict['MaxLengthEqual'],
                targets=torch.tensor([
                    -100,
                    1, 2, 2, 2, 3,
                    0, 0, 0, 0, 0, 0,
                    1, 2,
                    -100,
                ])
            ),
            standard_bieo_MaxLengthMore=dict(
                inputs=inputs_dict['MaxLengthMore'],
                targets=torch.tensor([
                    -100,
                    1, 2, 2, 2, 3,
                    0, 0, 0, 0, 0, 0,
                    1, 2,
                    -100,
                    -100, -100
                ])
            ),
            wordLevel_bieo_MaxLengthLess=dict(
                inputs=inputs_dict['MaxLengthLess'],
                targets=torch.tensor([
                    -100,
                    1, 1, 2, 3,
                    -100
                ])
            ),
            wordLevel_bieo_MaxLengthEqual=dict(
                inputs=inputs_dict['MaxLengthEqual'],
                targets=torch.tensor([
                    -100,
                    1, 1, 2, 3, 3,
                    0, 0, 0, 0, 0, 0,
                    1, 1,
                    -100,
                ])
            ),
            wordLevel_bieo_MaxLengthMore=dict(
                inputs=inputs_dict['MaxLengthMore'],
                targets=torch.tensor([
                    -100,
                    1, 1, 2, 3, 3,
                    0, 0, 0, 0, 0, 0,
                    1, 1,
                    -100,
                    -100, -100
                ])
            ),

            standard_bixo_MaxLengthLess=dict(
                inputs=inputs_dict['MaxLengthLess'],
                targets=torch.tensor([
                    -100,
                    2, 1, 3, 3,
                    -100
                ])
            ),
            standard_bixo_MaxLengthEqual=dict(
                inputs=inputs_dict['MaxLengthEqual'],
                targets=torch.tensor([
                    -100,
                    2, 1, 3, 3, 1,
                    0, 0, 0, 0, 0, 0,
                    2, 1,
                    -100,
                ])
            ),
            standard_bixo_MaxLengthMore=dict(
                inputs=inputs_dict['MaxLengthMore'],
                targets=torch.tensor([
                    -100,
                    2, 1, 3, 3, 1,
                    0, 0, 0, 0, 0, 0,
                    2, 1,
                    -100,
                    -100, -100
                ])
            ),
        )

        strategies = [
            'standard_io',
            'wordLevel_io',
            'standard_bio',
            'wordLevel_bio',
            'standard_bixo',
            'standard_bieo',
            'wordLevel_bieo'
        ]

        test_params = {
            '_'.join([strategy, max_length_key]): dict(
                df_label_map=df_label_map_dict[strategy.split('_')[1]],
                max_length=max_length_dict[max_length_key],
                strategy=strategy,
                df_text=df_text_dict[strategy.split('_')[1]],
                tokenizer=TokenizerMock(),
                is_train=False
            ) for (max_length_key, strategy) in product(max_length_dict, strategies)
        }

        for configuration, expected_output in expected_outputs.items():
            param = test_params[configuration]
            dataset = ArgumentMiningDataset(**param)
            inputs, targets = dataset[0]

            expected_inputs = expected_output['inputs']
            expected_targets = expected_output['targets']
            assert (targets == expected_targets).all(), f'{configuration} Targets:\n' \
                                                        f'Expected: {expected_targets}\n' \
                                                        f'Predicted: {targets}\n'

            for key in expected_inputs:
                assert (inputs[key] == expected_inputs[key]).all(), f'{configuration} {key}:\n' \
                                                                    f'Expected: {expected_inputs[key]}\n' \
                                                                    f'Predicted: {inputs[key]}\n'