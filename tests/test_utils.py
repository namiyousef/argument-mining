import unittest

from argminer.utils import get_predStr, _get_label_maps
import pandas as pd
from pandas.testing import assert_frame_equal
from itertools import product

class TestUtils(unittest.TestCase):

    def test_get_prediction_string(self):

        texts = [
                'Hi my name is Yousef',
                'Hi my\n name is Yousef',
                'Hi my name\n is Yousef',
                ' Hi  my name is Joe' # extra space example
            ]
        df_input = pd.DataFrame(dict(
            text=texts,
        )).assign(doc_id=list(range(len(texts))), label='Other')

        df_expected = df_input.copy().assign(predictionString=[
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4]
        ])

        df_output = get_predStr(df_input)

        assert_frame_equal(df_expected, df_output)

    def test_get_label_maps(self):
        strategies = ['io', 'bio', 'bieo', 'bixo']
        unique_labels_dict = dict(
            TUDarmstadt=['MajorClaim', 'Claim', 'Premise'],
            Persuade=['Lead', 'Position', 'Claim', 'Counterclaim', 'Rebuttal', 'Evidence', 'Concluding Statement']
        )

        test_params = {
            f'{dataset}_{strategy}': dict(
                unique_labels=unique_labels_dict[dataset], strategy=strategy
            ) for (dataset, strategy) in product(unique_labels_dict, strategies)
        }

        # -- test

        expected_outputs = dict(
            TUDarmstadt_io=pd.DataFrame(dict(
                label_id=[0, 1, 2, 3],
                label=['O', 'I-MajorClaim', 'I-Claim', 'I-Premise'],

            )),
            TUDarmstadt_bio=pd.DataFrame(dict(
                label_id=[0, 1, 2, 3, 4, 5, 6],
                label=['O', 'B-MajorClaim', 'I-MajorClaim', 'B-Claim', 'I-Claim', 'B-Premise', 'I-Premise'],

            )),
            TUDarmstadt_bieo=pd.DataFrame(dict(
                label_id=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                label=['O', 'B-MajorClaim', 'I-MajorClaim', 'E-MajorClaim',
                       'B-Claim', 'I-Claim', 'E-Claim', 'B-Premise', 'I-Premise', 'E-Premise'],

            )),
            TUDarmstadt_bixo=pd.DataFrame(dict(
                label_id=[0, 1, 2, 3, 4, 5, 6, 7],
                label=['O', 'X', 'B-MajorClaim', 'I-MajorClaim', 'B-Claim', 'I-Claim', 'B-Premise', 'I-Premise'],

            )),
            Persuade_io=pd.DataFrame(dict(
                label_id=[0, 1, 2, 3, 4, 5, 6, 7],
                label=[
                    'O',
                    'I-Lead', 'I-Position', 'I-Claim', 'I-Counterclaim',
                    'I-Rebuttal', 'I-Evidence', 'I-Concluding Statement'
                ],

            )),
            Persuade_bio=pd.DataFrame(dict(
                label_id=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
                label=[
                    'O',
                    'B-Lead', 'I-Lead', 'B-Position', 'I-Position', 'B-Claim', 'I-Claim',
                    'B-Counterclaim', 'I-Counterclaim', 'B-Rebuttal', 'I-Rebuttal',
                    'B-Evidence', 'I-Evidence', 'B-Concluding Statement', 'I-Concluding Statement'
                ],

            )),
            Persuade_bieo=pd.DataFrame(dict(
                label_id=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
                label=[
                    'O',
                    'B-Lead', 'I-Lead', 'E-Lead', 'B-Position', 'I-Position', 'E-Position',
                    'B-Claim', 'I-Claim', 'E-Claim', 'B-Counterclaim', 'I-Counterclaim', 'E-Counterclaim',
                    'B-Rebuttal', 'I-Rebuttal', 'E-Rebuttal', 'B-Evidence', 'I-Evidence', 'E-Evidence',
                    'B-Concluding Statement', 'I-Concluding Statement', 'E-Concluding Statement'
                ],
            )),
            Persuade_bixo=pd.DataFrame(dict(
                label_id=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                label=[
                    'O', 'X',
                    'B-Lead', 'I-Lead', 'B-Position', 'I-Position', 'B-Claim', 'I-Claim',
                    'B-Counterclaim', 'I-Counterclaim', 'B-Rebuttal', 'I-Rebuttal',
                    'B-Evidence', 'I-Evidence', 'B-Concluding Statement', 'I-Concluding Statement'
                ],
            ))
        )

        for configuration, expected_output in expected_outputs.items():
            param = test_params[configuration]
            output = _get_label_maps(**param)
            assert_frame_equal(output, expected_output)


