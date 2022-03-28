import unittest
from argminer.data import ArgumentMiningDataset
from argminer.utils import _get_label_maps
from transformers import AutoTokenizer
import pandas as pd
import torch


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
            #standard_bieo=0,
            #standard_bixo=[0]*7 + [2]*1 + [1]*1 + [3]*7 + [3]*2 + [4]*13 + [5]*3 + [6]*2 + [7]*9,
        )
        for i, strategy in enumerate(strategies):
            if strategy in expected_output:
                dataset = ArgumentMiningDataset(label_maps[strategy], text_dfs[strategy], tokenizer, max_length, strategy)
                for (inputs, targets) in dataset:
                    targets = targets[inputs['word_id_mask']]
                    expected_targets = torch.as_tensor(expected_output[strategy])
                    assert (targets == expected_targets).all(), f'assertion failed for strategy: {strategy}:\n{targets}\n{expected_targets}'
