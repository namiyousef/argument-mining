import unittest
import pandas as pd
from pandas.testing import assert_frame_equal
from itertools import product
from argminer.config import LABELS_MAP_DICT, PREDICTION_STRING_START_ID


class TestConfig(unittest.TestCase):

    def test_label_map_dict(self):

        # -- configuration
        datasets = ['TUDarmstadt', 'Persuade']
        labelling_schemes = ['io', 'bio', 'bieo', 'bixo']

        inputs = {
            f'{dataset}_{labelling_scheme}': (dataset, labelling_scheme) for dataset, labelling_scheme in product(
                datasets, labelling_schemes
            )
        }

        # -- test labelling schemes

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

        for configuration, (dataset, labelling_scheme) in inputs.items():
            expected_output = expected_outputs[configuration]
            output = LABELS_MAP_DICT[dataset][labelling_scheme]

            assert_frame_equal(expected_output, output)

    def test_constants(self):

        assert PREDICTION_STRING_START_ID == 0
        



if __name__ == '__main__':
    unittest.main()