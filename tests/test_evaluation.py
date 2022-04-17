import unittest
import torch
import pandas as pd
from pandas.testing import assert_frame_equal
from argminer.evaluation import get_word_labels, get_predictionString, evaluate

class TestEvaluation(unittest.TestCase):

    def test_get_word_labels(self):
        """
        Function to test the function 'get_word_labels'
        NOTE: this function is expected to be used in a pipeline. It can be
        used to concatenate sequences back into original form given IDs, usually used
        to concatenate subtokens back to text:
        Example: Yousef "Tsunami" Nami. I like NLP!
        Tokenized: Y ou sef "Tsu na mi" Na mi. I li ke NLP !
        IDs:       0  0   0    1  1  1   2  2  3  4  4   5 6

        Typically, this will have extra tokens that are to be ignored at the beginning
        and end of the sequence, there are indicated by -100.
        The final IDs is thus:
        -100, 0, 0, 0, 1, 1, 1, 2, 2, 3, 4, 4, 5, 6
        """

        # define word ids
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

        # define probabilities associated with each token for a 2 class problem
        probas = torch.tensor(
            [[
                [0.5, 0.2],
                [0.5, 0.1], [0.2, 0.7], [0.3, 0.1],
                [0.1, 0.8], [0.9, 0.7], [0.3, 0.1],
                [0.9, 0.8], [0.9, 0.2],
                [0.1, 0.8],
                [0.4, 0.2], [0.6, 0.7],
                [0.1, 0.1],
                [0.1, 0.1],
                [0.2, 0.8]
            ]],
        )

        # define the ground truth target classes
        classes = torch.tensor([
            [
                -100,
                1, 2, 3,
                1, 1, 1,
                1, 2,
                1,
                1, 2,
                1,
                1,
                -100
            ]
        ])


        # define expected targets with first aggregation strategy
        targets_first = [
            torch.tensor([[1, 1, 1, 1, 1, 1, 1]])
        ]


        # define expected outputs with first aggregation strategy
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

        # define expected outputs with mean aggregation strategy
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

        # define expected outputs with max agg strategy
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
                targets = get_word_labels(word_ids, classes, 'first', has_x)  # *** fixes targets to first aggregation strategy

                expected_output = expected_outputs[configuration]
                expected_targets = targets_first  # see above ***
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

        labels = [
            torch.tensor([1, 2, 2, 3, 3, 3, 4]),
            torch.tensor([1, 2, 3, 4, 5, 6, 7])
        ]
        doc_ids = torch.tensor([1, 2])

        df_expected = pd.DataFrame().from_records([
            (1, 1, {0, 1}),
            (1, 2, {2, 3, 4}),
            (1, 3, {5, 6, 7, 8}),
            (1, 4, {9, 10}),
            (2, 1, {0, 1}),
            (2, 2, {2, 3}),
            (2, 3, {4, 5}),
            (2, 4, {6, 7}),
            (2, 5, {8, 9}),
            (2, 6, {10, 11}),
            (2, 7, {12, 13}),
        ], columns=['id', 'class', 'predictionString'])
        df_output = get_predictionString(labels, doc_ids)

        # ensure equal for testing purposes
        df_output.predictionString = df_output.predictionString.apply(lambda x: sorted(x))
        df_expected.predictionString = df_expected.predictionString.apply(lambda x: sorted(x))

        assert_frame_equal(
            df_output,
            df_expected
        )

    def test_evaluate(self):
        # Gold sentence 1
        # Hi, this is an argument. -- Other (0)
        # According to Wikipedia, Wales is the largest country in the world. -- Evidence (1)
        # This is just filler.  -- Other (0)
        # Montenegro is beautiful. -- Claim (2)
        # In conclusion, -- Other (0)
        # Montenegro is the best country in the world. -- Concluding Statement (3)

        # Gold sentence 2
        # The new Batman film is a 6/10. -- Claim (2)
        # This is because, -- Other (0)
        # It is internally inconsistent with regards to the
        # assumptions it makes about the hero and decisions he takes. -- Evidence (1)

        
        df_targets_in = pd.DataFrame().from_records([
            (1, 0, {0, 1, 2, 3, 4}),
            (1, 1, {5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}),
            (1, 0, {16, 17, 18, 19}),
            (1, 2, {20, 21, 22}),
            (1, 0, {23, 24}),
            (1, 3, {25, 26, 27, 28, 29, 30, 31, 32}),

            (2, 2, {0, 1, 2, 3, 4, 5, 6}),
            (2, 0, {7, 8, 9}),
            (2, 1, {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27})
        ], columns=['id', 'class', 'predictionString'])

        df_outputs_in_dict = dict(
            EqualMisses=pd.DataFrame().from_records([
                (1, 0, {0, 1}), # FP
                (1, 1, {2, 3, 4, 5, 6, 7, 8, 9}), # FP
                (1, 0, {10, 11, 12, 13, 14, 15, 16, 17, 18, 19}), # FP
                (1, 2, {20, 21, 22, 23}), # should be a true positive
                (1, 0, {24, 25, 26, 27, 28, 29}), # FP
                (1, 3, {30, 31, 32}),  # FP
                (2, 2, {0, 1, 2}), # FP
                (2, 0, {7}), # FP
                (2, 1, {8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27}),
            ], columns=['id', 'class', 'predictionString']),
            AllTruePositive=pd.DataFrame().from_records([
                (1, 0, {0, 1, 2, 3, 4}),
                (1, 1, {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}),
                (1, 0, {15, 16, 17, 18}),
                (1, 2, {19, 20, 21}),
                (1, 0, {22, 23, 24}),
                (1, 3, {25, 26, 27, 28, 29, 30, 31, 32}),
                (2, 2, {0, 1, 2, 3}),
                (2, 0, {4, 5, 6, 7, 8, 9}),
                (2, 1, {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27})
            ], columns=['id', 'class', 'predictionString']),
            ComplexPrediction=pd.DataFrame().from_records([
                (1, 1, {0, 1, 2, 3, 4}), # prediction misses ground truth completely, FP1
                (1, 1, {5, 6, 7, 8, 9, 10}), #TP1
                (1, 0, {11, 12, 13}), #FP0
                (1, 1, {14, 15}), #FP1
                (1, 4, {16, 17}), # out of prediction value!, #FP4
                (1, 0, {18, 19}), #TP0
                (1, 2, {20, 21}), #TP2
                (1, 0, {22, 23}), #TP0
                (1, 3, {25, 26, 27, 28}), #TP3
                (1, 0, {29, 30}), # FP0
                (1, 3, {31, 32}), #FP3

                (2, 2, {0, 1, 2, 3}), #TP2
                (2, 0, {4, 5, 6}), #FP0
                (2, 1, {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25}), #TP1
                (2, 6, {26, 27}) #FP6
            ], columns=['id', 'class', 'predictionString'])
        )
        # TODO df_expected outputs will likely need to change if shape of
        # output scores changes!
        df_expected_outputs_dict = dict(
            AllTruePositive=pd.DataFrame({
                'class': [0, 1, 2, 3],
                'tp': [4, 2, 2, 1],
                'fn': [0, 0, 0, 0],
                'fp': [0, 0, 0, 0],
                'f1': [1., 1., 1., 1.] # TODO likely will remove
            }),
            EqualMisses=pd.DataFrame({
                'class': [0, 1, 2, 3],
                'tp': [0., 1., 1., 0.],
                'fn': [4., 1., 1., 1.],
                'fp': [4., 1., 1., 1.],
                'f1': [0, 0.5, 0.5, 0]
            }),
            ComplexPrediction=pd.DataFrame({
                'class': [0, 1, 2, 3, 4, 6],
                'tp': [2., 2., 2., 1., 0., 0.],
                'fn': [2., 0., 0., 0., 0., 0.],
                'fp': [3., 2., 0., 1., 1., 1.],
                'f1': [2/4.5, 2/3, 1, 1/1.5, 0, 0]
            })
        )

        for configuration, df_expected_output in df_expected_outputs_dict.items():
            df_output = evaluate(
                df_outputs_in_dict[configuration],
                df_targets_in
            ).sort_values('class').reset_index(drop=True)
            assert_frame_equal(df_output, df_expected_output, check_dtype=False)



    def test_inference(self):
        pass