import unittest
import torch
from argminer.evaluation import get_word_labels

class TestEvaluation(unittest.TestCase):

    def test_get_word_labels(self):

        # Example sentence and tokens
        # sentence: Yousef Mohammad Nami. I come from Dubai.
        # tokens: Y ou sef Mo ham mad Na mi. I co me from Du bai

        word_ids = torch.tensor([[
            -100,
            0, 0, 0,
            1, 1, 1,
            2, 2,
            3,
            4, 4,
            5,
            6,
            -100
        ]])

        # TODO not implemented in dataset yet. Just adding
        # for expected behaviour (in principle this should
        # be the same as calling first...)

        word_ids_ignore_subtokens = torch.tensor([
            [
                -100,
                0, -100, -100,
                1, -100, -100,
                2, -100,
                3,
                4, -100,
                5,
                6, -100,
                -100
            ]
        ])
        
        probas = torch.tensor(
            [[
                [0.5, 0.2], # ignored
                [0.5, 0.1], [0.2, 0.7], [0.3, 0.1], # for first 0, for mean 0, for max 1
                [0.1, 0.8], [0.9, 0.7], [0.3, 0.1], # for first 1, for mean 1, for max 0
                [0.9, 0.8], [0.9, 0.2], # for first= 0, mean=0, max=0
                [0.1, 0.8], # for first 0, for max = 1, for mean=1
                [0.4, 0.2], [0.6, 0.7],  # first = 1, mean = 0, max = 1
                [0.1, 0.1],  # first = , mean = , max =
                [0.1, 0.1],
                [0.2, 0.8]  # ignored
            ]],
        )

        # thing about how to test for targets
        # TODO note that for targets
        # you will have mean and agg affecting it as well
        # this is unintended behaviour
        classes = torch.tensor([
            [
                -100,
                1, 2, 3, # case bieo
                1, 1, 1, # case word level
                1, 2, # case bio
                1,
                1, 2,
                1,
                1, 1
                -100
            ]
        ])


        # change output type
        targets_first = [
            torch.tensor([[1, 1, 1, 1, 1, 1, 1]])
        ]
        outputs_first = [
            torch.tensor([
                [0.5, 0.1],
                [0.1, 0.8],
                [0.9, 0.8],
                [0.1, 0.8],
                [0.4, 0.2],
                [0.1, 0.1],
                [0.1, 0.1],
            ])

        ]

        # TODO note: always sets targets to "first"
        # in the tests. This assumes that the user
        # knows that they are doing. Need to have integration
        # test to prevent mistake from happening,
        # also add to DOCS!
        outputs_mean = [
            torch.tensor([
                [1 / 3, 0.3],
                [13 / 30, 8 / 15],
                [0.9, 0.5],
                [0.1, 0.8],
                [0.5, 0.45],
                [0.1, 0.1],
                [0.1, 0.1]
            ])

        ]

        outputs_max = [
            torch.tensor([
                [0.5, 0.7],
                [0.9, 0.8],
                [0.9, 0.8],
                [0.1, 0.8],
                [0.6, 0.7],
                [0.1, 0.1],
                [0.1, 0.1],
            ])
        ]

        # todo integration test. Make sure that targets
        # cannot be passed with non-first!
        expected_outputs = dict(
            first_HasX=outputs_first,
            mean_HasX=outputs_first,
            max_HasX=outputs_first,
            first_NoX=outputs_first,
            mean_NoX=outputs_mean,
            max_NoX=outputs_max,
        )

        for agg_strategy in ['first', 'mean', 'max']:
            for identifier, has_x in {'HasX': True, 'NoX': False}.items():
                configuration = f'{agg_strategy}_{identifier}'
                outputs = get_word_labels(word_ids, probas, agg_strategy, has_x)
                targets = get_word_labels(word_ids, classes, 'first', has_x) # TODO see ***

                expected_output = expected_outputs[configuration]
                expected_targets = targets_first # TODO see ***
                assert torch.tensor([
                    torch.allclose(output, expected_output_) for output, expected_output_ in zip(
                        outputs, expected_output
                    )
                    ]).all(), f'For {configuration} outputs FAILED\n' \
                                                   f'EXPECTED: {expected_output}\n' \
                                                   f'PREDICTED: {outputs}\n'
                assert torch.tensor([
                    (target == expected_target).all() for target, expected_target in zip(
                        targets, expected_targets
                    )
                    ]).all(), f'For {configuration} targets FAILED\n' \
                                                    f'EXPECTED: {expected_targets}\n' \
                                                    f'PREDICTED: {targets}'
    def test_get_predictionString(self):
        pass

    def test_evaluate(self):
        pass

    def test_inference(self):
        pass