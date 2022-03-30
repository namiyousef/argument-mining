import torch


def test(model, testloader, collect_predictions=False):

    model.eval()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(testloader):
            # TODO need to extract relevant items from inputs
            # TODO move inputs and model to relevant devices
            loss, outputs = model(
                labels=targets,
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                return_dict=False
            )
            if collect_predictions:
                pass

    return predictions


def get_word_labels(inputs, outputs, agg_strategy, has_x):
    """
    Function that gets the labels for words from it's subtoken given an aggregation strategy
    :param model:
    :param outputs:
    :param word_ids:
    :param agg_strategy:
    :return:
    """
    pred_labels = []
    for (predictions, word_ids) in zip(outputs, inputs):
        mask = word_ids != -1 # TODO double check this across code
        word_ids = word_ids[mask]
        predictions = predictions[mask]

        unique_word_ids, word_id_counts = torch.unique_consecutive(word_ids, return_counts=True)
        agg_predictions = torch.zeros(
            (unique_word_ids.shape[0], predictions.shape[-1]),
            dtype=predictions.dtype
        )

        start_id = 0
        for i, (unique_word_id, word_id_count) in enumerate(zip(unique_word_ids, word_id_counts)):
            end_id = start_id + word_id_count
            prediction_slice = predictions[start_id: end_id]

            if agg_strategy == 'mean' and not has_x:
                agg_predictions[i] = prediction_slice.mean(dim=0)
            elif agg_strategy == 'max' and not has_x:
                agg_predictions[i] = prediction_slice.max(dim=0)
            elif agg_strategy == 'first':
                agg_predictions[i] = prediction_slice[0]
            start_id = end_id

        # TODO need options to return the probas and also the assoc labels?
        # maybe better to go in df form here?
        pred_labels.append(agg_predictions)
        
    return pred_labels

def get_labels(tensor, return_type):
    probas, indices = torch.max(tensor, 1)
    if return_type == 'labels':
        return indices
    elif return_type == 'probas':
        return probas
    elif return_type == 'both':
        return (probas, indices)

class Inference:

    def __init__(self, collect_predictions, agg, return_type):
        self.collect_predictions = collect_predictions
        self.agg = agg
        self.return_type = return_type

    def preprocess(self):
        pass

    def process(self):
        pass

    def postprocess(self):
        pass

    def __call__(self, output_tensor):
        pass
    
def inference(outputs, agg=False):
    """
    Function that aggregates
    :param model:
    :param outputs:
    :param word_ids:
    :param agg_strategy:
    :return:
    """
    if agg:
        pass


    pass

"""
Steps of inference:
- predict using the trained model, this is done in batches
- aggregate the batches into words, this will lead to differently sized predictions
- using the aggregated predictions, compare against the input data with predictionString
"""


def evaluate(df_outputs, df_targets):
    # TODO problem of duplicates? Think about this with test cases
    """
    Calculates the F-Score given prediction strings
    """
    # -- Constants
    gt, pred = '_gt', '_pred'

    #    Merge constants
    merge_on = ['id', 'class']
    suffixes = (gt, pred)

    #    Columns labels
    overlap_pred_gt = f'overlap{pred}{gt}'  # how much of ground truth in prediction
    overlap_gt_pred = f'overlap{gt}{pred}'  # how much of prediction in ground truth
    pred_col_name = 'predictionString'
    gt = f'{pred_col_name}{gt}'
    pred = f'{pred_col_name}{pred}'

    # TODO is this even correct?? Would you not be repeating items??
    # find all combinations
    # REally really not sure of this assumption...
    df_targets = df_targets.merge(
        df_outputs, on=merge_on, suffixes=suffixes
    )
    # find intersection and normalise against each item
    df_targets[overlap_pred_gt] = df_targets[[gt, pred]].apply(
        lambda x: len(x[gt].intersection(x[pred])) / len(x[pred]), axis=1
    )

    df_targets[overlap_gt_pred] = df_targets[[pred, gt]].apply(
        lambda x: len(x[gt].intersection(x[pred])) / len(x[gt]), axis=1
    )

    df_targets['tp'] = df_targets[[overlap_pred_gt, overlap_gt_pred]].apply(
        lambda x: int(x.overlap_pred_gt >= 0.5 and x.overlap_gt_pred >= 0.5), axis=1
    )
    df_targets['overlap_sum'] = df_targets.overlap_pred_gt + df_targets.overlap_gt_pred
    df_targets[pred] = df_targets[pred].apply(lambda x: tuple(x))
    df_targets[gt] = df_targets[gt].apply(lambda x: tuple(x))

    # only keep the ones with the highest overall score
    df_targets_match = df_targets[df_targets['tp'] == 1].groupby(
        ['id', 'class', pred]
    ).agg({'overlap_sum': 'max', gt: lambda x: x})

    # need to find TP by class!
    TP = df_targets_match.shape[0]

    # find the items that have no match, but also do not have predictionStrings that are matched
    df_targets_no_match = df_targets[
        (df_targets['tp'] == 0) & (~df_targets.set_index(['id', 'class', pred]).index.isin(df_targets_match.index))
        ]

    ids_no_matches = (df_targets_no_match.overlap_pred_gt == 0) & (df_targets_no_match.overlap_gt_pred == 0)
    # filter out the false positives
    df_false_positives = df_targets_no_match[
        ~ids_no_matches
    ]

    df_false_negatives = df_targets_no_match[
        (ids_no_matches) & (~df_targets_no_match.set_index(['id', 'class', gt]).index.isin(
            df_targets_match.reset_index().set_index(['id', 'class', gt]).index
        ))
        ]

    TP = df_targets_match.reset_index().groupby(['id', 'class']).size().to_frame('tp').reset_index()
    FN = df_false_negatives.groupby(['id', 'class']).size().to_frame('fn').reset_index()
    FP = df_false_positives.groupby(['id', 'class']).size().to_frame('fp').reset_index()

    scores = TP.merge(
        FN.merge(FP, how='outer'), how='outer'
    ).fillna(0)

    # TODO need to apply f1 score
    return scores