import torch
import pandas as pd

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

def get_predictionString(labels, doc_ids):
    """
    """
    # TODO need to ensure that we are mapping to the correct class names and doc IDs!
    ids = []
    classes = []
    prediction_strings = []
    for doc_id, label in zip(doc_ids, labels):
        unique_classes, unique_class_counts = torch.unique_consecutive(label, return_counts=True)
        start_id = 1
        for unique_class, unique_class_count in zip(unique_classes, unique_class_counts):
            ids.append(doc_id.item())
            classes.append(unique_class.item())
            end_id = start_id + unique_class_count + 1
            prediction_strings.append(set(range(start_id, end_id)))
            start_id = end_id
    return pd.DataFrame(data={'id': ids, 'class': classes, 'predictionString': prediction_strings})

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
    gt_id = f'index{gt}'
    pred_id = f'index{pred}'

    overlap_pred_gt = f'overlap{pred}{gt}'  # how much of ground truth in prediction
    overlap_gt_pred = f'overlap{gt}{pred}'  # how much of prediction in ground truth
    pred_col_name = 'predictionString'
    gt = f'{pred_col_name}{gt}'
    pred = f'{pred_col_name}{pred}'

    # TODO is this even correct?? Would you not be repeating items??
    # find all combinations
    # REally really not sure of this assumption..

    df_targets = df_targets.reset_index()
    df_outputs = df_outputs.reset_index()

    df_targets = df_targets.merge(
        df_outputs, on=merge_on, how='outer', suffixes=suffixes
    )

    df_targets[[gt, pred]] = df_targets[[gt, pred]].where(~df_targets[[gt, pred]].isnull(), {-1})

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
    df_targets['overlap_max'] = df_targets[[overlap_pred_gt, overlap_gt_pred]].max(axis=1)

    df_targets[pred] = df_targets[pred].apply(lambda x: tuple(x))
    df_targets[gt] = df_targets[gt].apply(lambda x: tuple(x))

    # only keep the ones with the highest overall score

    # I'm not convinced by this tbh? What about cases of perfect match on one but not on the other?
    # e.g. case when you have 0.5 and 1 and then 0.8 and 1
    df_targets_match = df_targets[df_targets['tp'] == 1].sort_values('overlap_max', ascending=False).groupby(
        ['id', 'class', gt]  # group by pred or gt?
    ).first()

    TP = df_targets_match.groupby('class')[pred_id].nunique().to_frame('tp').reset_index()

    df_false_positive = df_targets[
        # they didn't do this filter?
        # (df_targets['tp'] == 0) &
        (~df_targets.set_index(pred_id).index.isin(df_targets_match[pred_id]))

    ]

    FP = df_false_positive.groupby('class')[pred_id].nunique().to_frame('fp').reset_index()

    matched_gt_id = df_targets[df_targets['tp'] == 1][gt_id].unique()

    print(df_targets[df_targets[gt_id].isin(matched_gt_id)].sort_values('class'))

    df_false_negative = df_targets[
        # (df_targets['tp'] == 0) &
        (~df_targets.set_index(gt_id).index.isin(matched_gt_id))
    ]

    FN = df_false_negative.groupby('class')[gt_id].nunique().to_frame('fn').reset_index()

    # TP = df_targets_match.reset_index().groupby(['id', 'class']).size().to_frame('tp').reset_index()
    # FN = df_false_negatives.groupby(['id', 'class']).size().to_frame('fn').reset_index()
    # FP = df_false_positives.groupby(['id', 'class']).size().to_frame('fp').reset_index()

    scores = TP.merge(
        FN.merge(FP, how='outer'), how='outer'
    ).fillna(0)
    scores = scores.assign(f1=lambda x: x['tp'] / (x['tp'] + 1 / 2 * (x['fp'] + x['fn'])))
    print(scores)

    return scores['f1'].mean()