# -- public imports
import io
import json
from pandas.testing import assert_frame_equal
import unittest
import sys
import pandas as pd
import numpy as np
from itertools import permutations
from unittest.mock import Mock, patch

# -- private imports
from argminer.data import DataProcessor, TUDarmstadtProcessor, PersuadeProcessor

# -- functions
class Logger:
    def __init__(self):
        print(f'Running test "{sys._getframe(1).f_code.co_name}"')
        print("=" * 50)
        self.counter = 1

    def log(self, description):
        print(f'{self.counter}: {description}')
        self.counter += 1

class TestDataProcessor(unittest.TestCase):

    def test_base_processor(self):
        logger = Logger()

        logger.log('Test the preproc, proc, postproc cannot be called in incorrect order')

        # define dummy subclass to test superclass features
        class DummyProcessor(DataProcessor):
            def __init__(self, path=''):
                super().__init__(path)
            def _preprocess(self):

                self.dataframe = pd.DataFrame().from_records([
                    (1, '1', '1'),
                    (1, '2', '2'),
                    (1, '3', '3'),
                    (1, '4', '4'),
                ])
                self.status = 'preprocessed'

            def _process(self):
                pass

            def _postprocess(self):
                pass

        processor = DummyProcessor()

        status = ['preprocess', 'process', 'postprocess']
        status_perm = list(permutations(status, r=2))

        for s1, s2 in status_perm:
            processor.status = s1
            try:
                getattr(processor, s2)
                raise Exception(f'Failure: Successfully ran {s2} after {s1}')
            except:
                pass
            processor.status = None

        logger.log('Test saving')

        logger.log('Test loading')


        logger.log('Test default train test split')




        logger.log('Test custom train test split')


    def test_tu_darmstadt_processor(self):

        df_tu_darmstadt_raw = pd.DataFrame.from_records([
                ('1', 'Sentence 1', '1'),
                ('1', 'Sentence 2', '3'),
                ('1', 'Sentence 3', '2'),
                ('2', 'Sentence 1', '1'),
                ('3', 'Sentence 1', '0'),
                ('3', 'Sentence 2', '1'),
            ], columns=['doc_id', 'text', 'label'])

        def _preprocess_mock(self):
            self.dataframe = df_tu_darmstadt_raw
            self.status = 'preprocessed'
            return self

        def _save_json_mock(self):
            return self.dataframe

        df_expected_outputs = dict(
            test=pd.DataFrame.from_records([
                ('2', 'Sentence 1', ['B-1', 'I-1'], [0, 1]),
                ('3', 'Sentence 1', ['B-0', 'I-0'], [0, 1]),
                ('3', 'Sentence 2', ['B-1', 'I-1'], [2, 3]),
            ], columns=['doc_id', 'text', 'label', 'predictionString']),
            train=pd.DataFrame.from_records([
                ('1', 'Sentence 1', ['B-1', 'I-1'], [0, 1]),
                ('1', 'Sentence 2', ['B-3', 'I-3'], [2, 3]),
                ('1', 'Sentence 3', ['B-2', 'I-2'], [4, 5]),
            ], columns=['doc_id', 'text', 'label', 'predictionString']),
        )

        df_expected_outputs_post = dict(
            test=pd.DataFrame.from_records([
                ('2', 'Sentence 1', ['B-1', 'I-1'], [0, 1]),
                ('3', 'Sentence 1 Sentence 2', ['B-0', 'I-0', 'B-1', 'I-1'], [0, 1, 2, 3]),
            ], columns=['doc_id', 'text', 'labels', 'predictionString']),
            train=pd.DataFrame.from_records([
                ('1', 'Sentence 1 Sentence 2 Sentence 3', ['B-1', 'I-1', 'B-3', 'I-3', 'B-2', 'I-2'], [0, 1, 2, 3, 4, 5]),
            ], columns=['doc_id', 'text', 'labels', 'predictionString']),
        )

        logger = Logger()

        logger.log('Test train test split - during process method')
        
        with patch.object(
                TUDarmstadtProcessor, '_preprocess', _preprocess_mock
        ) as _preprocess_mock, \
                patch.object(pd, 'read_csv') as read_csv_mock,\
                patch.object(DataProcessor, 'save_json', _save_json_mock) as _save_json_mock:
            read_csv_mock.return_value = pd.DataFrame().from_records([
                ('1', 'TRAIN'),
                ('2', 'TEST'),
                ('3', 'TEST')
            ], columns=['ID', 'SET'])
            # TODO needs to add a test for the validation....
            for split in ['test', 'train']:
                processor = TUDarmstadtProcessor().preprocess().process('bio', split=split)
                df_output = processor.dataframe.reset_index(drop=True)
                assert_frame_equal(
                    df_output,
                    df_expected_outputs[split]
                )


                processor = processor.postprocess()
                df_output = processor.dataframe.reset_index(drop=True)[['doc_id', 'text', 'labels', 'predictionString']]
                assert_frame_equal(
                    df_output,
                    df_expected_outputs_post[split][['doc_id', 'text', 'labels', 'predictionString']]
                )


    def test_persuade_processor(self):


        logger = Logger()

        df_persuade_raw = pd.DataFrame.from_records([
            ('1', 'Sentence 1', '1'),
            ('1', 'Sentence 2', '3'),
            ('1', 'Sentence 3', '2'),
            ('2', 'Sentence 1', '1'),
            ('3', 'Sentence 1', '0'),
            ('3', 'Sentence 2', '1'),
        ], columns=['doc_id', 'text', 'label'])

        def _preprocess_mock(self):
            self.dataframe = df_persuade_raw
            self.status = 'preprocessed'
            return self

        def _save_json_mock(self):
            return self.dataframe

        df_expected_outputs = dict(
            train=pd.DataFrame.from_records([
                ('2', 'Sentence 1', ['B-1', 'I-1'], [0, 1]),
                ('3', 'Sentence 1', ['B-0', 'I-0'], [0, 1]),
                ('3', 'Sentence 2', ['B-1', 'I-1'], [2, 3]),
            ], columns=['doc_id', 'text', 'label', 'predictionString']),
            test=pd.DataFrame.from_records([
                ('1', 'Sentence 1', ['B-1', 'I-1'], [0, 1]),
                ('1', 'Sentence 2', ['B-3', 'I-3'], [2, 3]),
                ('1', 'Sentence 3', ['B-2', 'I-2'], [4, 5]),
            ], columns=['doc_id', 'text', 'label', 'predictionString']),
        )

        df_expected_outputs_post = dict(
            train=pd.DataFrame.from_records([
                ('2', 'Sentence 1', ['B-1', 'I-1'], [0, 1]),
                ('3', 'Sentence 1 Sentence 2', ['B-0', 'I-0', 'B-1', 'I-1'], [0, 1, 2, 3]),
            ], columns=['doc_id', 'text', 'labels', 'predictionString']),
            test=pd.DataFrame.from_records([
                ('1', 'Sentence 1 Sentence 2 Sentence 3', ['B-1', 'I-1', 'B-3', 'I-3', 'B-2', 'I-2'],
                 [0, 1, 2, 3, 4, 5]),
            ], columns=['doc_id', 'text', 'labels', 'predictionString']),
        )



        logger.log('Test train test split - during process method')

        with patch.object(
                PersuadeProcessor, '_preprocess', _preprocess_mock
        ) as _preprocess_mock, \
                patch.object(np.random, 'shuffle') as shuffle_mock, \
                patch.object(DataProcessor, 'save_json', _save_json_mock) as _save_json_mock:
            shuffle_mock.return_value = None
            # TODO needs to add a test for the validation....
            for split in ['test', 'train']:
                processor = PersuadeProcessor().preprocess().process('bio', split=split, test_size=0.5)
                df_output = processor.dataframe.reset_index(drop=True)
                assert_frame_equal(
                    df_output,
                    df_expected_outputs[split]
                )
                processor = processor.postprocess()
                df_output = processor.dataframe.reset_index(drop=True)
                assert_frame_equal(
                    df_output[['doc_id', 'text', 'labels', 'predictionString']],
                    df_expected_outputs_post[split][['doc_id', 'text', 'labels', 'predictionString']]
                )

    def test_arg2020_processor(self):
        pass


if __name__ == '__main__':
    unittest.main()