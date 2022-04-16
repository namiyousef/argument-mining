import unittest
from itertools import product

from argminer.data import _generate_entity_labels


class TestDataHelpers(unittest.TestCase):

    def test_generate_entity_labels(self):
        # -- configuration
        labels_dict = dict(
            Other='Other',
            NotOther='NotOther'
        )
        add_beg_dict = dict(Beg=True, NotBeg=False)
        add_end_dict = dict(End=True, NotEnd=False)

        test_params = {}
        for (label, add_beg_key, add_end_key) in product(labels_dict, add_beg_dict, add_end_dict):
            identifier = f'{label}{add_beg_key}{add_end_key}'
            test_params[identifier] = dict(
                label=labels_dict[label],
                add_beg=add_beg_dict[add_beg_key],
                add_end=add_end_dict[add_end_key]
            )

        # -- test normal case
        length = 3

        expected_outputs = dict(
            OtherBegEnd=['O', 'O', 'O'],
            OtherBegNotEnd=['O', 'O', 'O'],
            OtherNotBegEnd=['O', 'O', 'O'],
            OtherNotBegNotEnd=['O', 'O', 'O'],
            NotOtherBegEnd=['B-NotOther', 'I-NotOther', 'E-NotOther'],
            NotOtherBegNotEnd=['B-NotOther', 'I-NotOther', 'I-NotOther'],
            NotOtherNotBegEnd=['I-NotOther', 'I-NotOther', 'E-NotOther'],
            NotOtherNotBegNotEnd=['I-NotOther', 'I-NotOther', 'I-NotOther'],
        )

        for configuration, expected_output in expected_outputs.items():
            params = test_params[configuration]
            output = _generate_entity_labels(length, **params)
            assert expected_output == output

        # test edge case (even)
        length = 2

        expected_outputs = dict(
            NotOtherBegEnd=['B-NotOther', 'E-NotOther']
        )

        for configuration, expected_output in expected_outputs.items():
            params = test_params[configuration]
            output = _generate_entity_labels(length, **params)
            assert expected_output == output

        # -- test edge case (only 1)
        length = 1

        expected_outputs = dict(
            NotOtherBegEnd=['B-NotOther']
        )

        for configuration, expected_output in expected_outputs.items():
            params = test_params[configuration]
            output = _generate_entity_labels(length, **params)
            assert expected_output == output

