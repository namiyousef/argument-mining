# -- public imports
import torch
import pandas as pd

# -- private imports
from colabtools.config import DEVICE
from colabtools.utils import move_to_device

def get_word_labels(inputs, outputs, agg_strategy, has_x):
    """
    Function that aggregates from subtokens back to words given an aggregation strategy,
    probabilities for each subtoken, and word_ids

    :param inputs: word ids that map each subtoken to it's word. This comes from tokenizer(string).word_ids()
    but has the None parameters replaced with '-1' indicating that it is not a word (e.g. SEP, CLS)
    :type inputs: torch.Tensor
    :param outputs: raw predictions from a model output (OR the target variables)
    :type outputs: torch.Tensor
    :param agg_strategy: defines how the subtokens are aggregated back to words
    :type agg_strategy: str

    :returns: list of shortened tensors corresponding to each word
    :rtype: list
    """
    pred_labels = []
    for (predictions, word_ids) in zip(outputs, inputs):
        mask = word_ids != -1 # TODO double check this across code

        # filter out SEP, CLS
        word_ids = word_ids[mask]
        predictions = predictions[mask]

        unique_word_ids, word_id_counts = torch.unique_consecutive(word_ids, return_counts=True)
        agg_predictions = torch.zeros(
            (
                unique_word_ids.shape[0],
                # TODO below len() is a hotfix to enable dual behaviour for raw probabilities (e.g. outputs) and targets
                predictions.shape[-1] if len(predictions.shape) > 1 else 1
            ),
            dtype=predictions.dtype
        )

        start_id = 0
        for i, (unique_word_id, word_id_count) in enumerate(zip(unique_word_ids, word_id_counts)):

            end_id = start_id + word_id_count
            # get segments corresponding to word
            prediction_slice = predictions[start_id: end_id]

            # apply aggregation strategy
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
    Function that takes list of tensors along with unique document ids to generate a dataframe
    with predictionStrings for each (doc_id, class).
    :param labels: list of tensors that contain classes
    :type labels: list
    :param doc_ids: collection of document ids pertaining to each item in labels
    :type doc_ids: torch.Tensor
    :returns: dataframe in the following form (doc_id, class, predictionString) where predictionString is a set
    """
    # TODO need to ensure that we are mapping to the correct class names and doc IDs!

    ids = []
    classes = []
    prediction_strings = []

    for doc_id, label in zip(doc_ids, labels):
        unique_classes, unique_class_counts = torch.unique_consecutive(label, return_counts=True)
        start_id = 1 # TODO this defines the start of a predictionString
        for unique_class, unique_class_count in zip(unique_classes, unique_class_counts):
            ids.append(doc_id.item())
            classes.append(unique_class.item())
            end_id = start_id + unique_class_count + 1
            prediction_strings.append(set(range(start_id, end_id)))
            start_id = end_id
    return pd.DataFrame(data={'id': ids, 'class': classes, 'predictionString': prediction_strings})

def evaluate(df_outputs, df_targets):
    """
    Calculates the macro f1 score for a given batch of data
    This function is based on the instructions from the Kaggle evaluation as well as
    the notebook from Changmao's friend

    # TODO rename this function
    :param df_outputs: outputs dataframe directly from get_predictionString
    :type df_outputs: pd.DataFrame
    :param df_targets: targets dataframe directly from get_predictionString
    :type df_targets: pd.DataFrame

    :returns: scores for each class (tp, fp, fn and macro f1)
    :rtype: pd.DataFrames
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
    gt = f'{pred_col_name}{gt}'  # column name for ground truth prediction string
    pred = f'{pred_col_name}{pred}'  # column name for predicted prediction string

    # get unique ids for each row
    df_targets = df_targets.reset_index()
    df_outputs = df_outputs.reset_index()

    # merge on all combinations
    df_targets = df_targets.merge(
        df_outputs, on=merge_on, how='outer', suffixes=suffixes
    )

    # replace null values with predictionString set with -1 (so that it does not affect overlap calc)
    # NOTE: the .nan of the 'index_' columns not filled.
    df_targets[[gt, pred]] = df_targets[[gt, pred]].where(~df_targets[[gt, pred]].isnull(), {-1})

    # find intersection and normalise against each item
    df_targets[overlap_pred_gt] = df_targets[[gt, pred]].apply(
        lambda x: len(x[gt].intersection(x[pred])) / len(x[pred]), axis=1
    )
    df_targets[overlap_gt_pred] = df_targets[[pred, gt]].apply(
        lambda x: len(x[gt].intersection(x[pred])) / len(x[gt]), axis=1
    )

    # label true positives
    df_targets['tp'] = df_targets[[overlap_pred_gt, overlap_gt_pred]].apply(
        lambda x: int(x.overlap_pred_gt >= 0.5 and x.overlap_gt_pred >= 0.5), axis=1
    )
    # find the maximum overlap
    # NOTE: I'm not convinced about this. There will be cases where this does not give a correct answer
    # for example: you can have overlap_gt_pred = [0.5, 0.8] and overlap_pred_gt = [1, 1]
    # the max operation will give overlap_max = [1, 1] and after group by you will be left with
    # the first example, which had a lower probability on the overlap_gt_pred!

    df_targets['overlap_max'] = df_targets[[overlap_pred_gt, overlap_gt_pred]].max(axis=1)

    # fix data types for grouping (sets are not hashable)
    df_targets[pred] = df_targets[pred].apply(lambda x: tuple(x))
    df_targets[gt] = df_targets[gt].apply(lambda x: tuple(x))

    # Group by and take the maximum in caes of more than one match (see above note for CAVEAT!)
    df_targets_match = df_targets[df_targets['tp'] == 1].sort_values('overlap_max', ascending=False).groupby(
        ['id', 'class', gt]  # TODO group by pred or gt?
    ).first()


    # get false positives as prediction instances that don't appear in the true positives
    df_false_positive = df_targets[
        # TODO changmao's friend does not apply this condition, so I've commented it
        # (df_targets['tp'] == 0) &
        (~df_targets.set_index(pred_id).index.isin(df_targets_match[pred_id]))

    ]

    # get false negatives as gt instances that don't appear in the true positives
    matched_gt_id = df_targets[df_targets['tp'] == 1][gt_id].unique()

    df_false_negative = df_targets[
        # TODO changmao's friend does not apply this condition, so I've commented it
        # (df_targets['tp'] == 0) &
        (~df_targets.set_index(gt_id).index.isin(matched_gt_id))
    ]

    # get score counts by grouping per class
    TP = df_targets_match.groupby('class')[pred_id].nunique().to_frame('tp').reset_index()
    FP = df_false_positive.groupby('class')[pred_id].nunique().to_frame('fp').reset_index()
    FN = df_false_negative.groupby('class')[gt_id].nunique().to_frame('fn').reset_index()

    # merge and fill empty ones with 0
    scores = TP.merge(
        FN.merge(FP, how='outer'), how='outer'
    ).fillna(0)

    # calculate macro_f1 score
    scores = scores.assign(f1=lambda x: x['tp'] / (x['tp'] + 1 / 2 * (x['fp'] + x['fn'])))

    return scores

def inference(model, testloader, metrics=[]):
    # TODO add options for agg method
    """
    Takes a trained model and evaluates its performance based on the Macro F1 score from Kaggle as well
    as custom metrics

    :param model: trained transformers model
    :type model: transformers.models.*
    :param testloader: data loader based on custom dataset
    :type testloader: torch.utils.data.DataLoader
    :param metrics: list of metrics to monitor model performance
    :type metrics: list
    """

    # set model to test mode, set correct device
    model.eval(); model.to(DEVICE)

    # get mapping to move from positional labels to core labels, e.g. 'B-{type}' becomes '{type}'
    reduce_map = testloader.dataset.reduce_map
    reduce_map_values = torch.as_tensor(list(reduce_map.values()))
    reduce_map_values = move_to_device(reduce_map_values, DEVICE)

    total_metrics = []
    total_scores = []

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(testloader):

            inputs = move_to_device(inputs, DEVICE)
            targets = move_to_device(targets, DEVICE)

            loss, outputs = model(
                labels=targets,
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                return_dict=False
            )
            word_ids = inputs['word_ids']
            doc_ids = inputs['index']

            # measure performance at subtoken level
            df_metrics_no_agg = pd.DataFrame.from_records({
                f'{metric.__class__.__name__}_no_agg': metric(outputs, targets) for metric in metrics
            })

            # aggregate from subtoken to words # TODO strat needs to be parametrised
            word_label_probas = get_word_labels(word_ids, outputs, agg_strategy='first', has_x=False)
            word_label_ids = [tensor.argmax(dim=1) for tensor in word_label_probas]

            # aggregate from subtoken to words for targets. Note: agg_strategy always first
            target_label_probas = get_word_labels(word_ids, targets, agg_strategy='first', has_x=False)
            # TODO this is a hotfix. Need an automatic dimensioning tool!
            target_label_ids = [tensor.flatten() for tensor in target_label_probas]


            # measure performance at word level, before labels are mapped to reduced form
            df_metrics_agg = pd.DataFrame.from_records({
                f'{metric.__class__.__name__}_agg': [
                    metric(output, target).item() for output, target in zip(word_label_ids, target_label_ids)
                ] for metric in metrics
            })

            # map word labels to reduced form
            word_labels = [reduce_map_values[label_ids] for label_ids in word_label_ids]
            target_labels = [reduce_map_values[label_ids] for label_ids in target_label_ids]


            # measure performance at word level, for reduced labels
            df_metrics_agg_reduced = pd.DataFrame.from_records({
                f'{metric.__class__.__name__}_agg_reduced': [
                    metric(output, target).item() for output, target in zip(word_labels, target_labels)
                ] for metric in metrics
            })

            # combine all metrics
            df_metrics = pd.concat([df_metrics_no_agg, df_metrics_agg, df_metrics_agg_reduced], axis=1)


            # get dataframes of (doc_id, class, prediction_string)
            df_targets_predString = get_predictionString(target_labels, doc_ids)
            df_outputs_predString = get_predictionString(word_labels, doc_ids)

            # get scores (and macro f1)
            df_scores = evaluate(df_outputs_predString, df_targets_predString)

            total_metrics.append(df_metrics)
            total_scores.append(df_scores)

    df_metrics_total = pd.concat(total_metrics)
    df_scores_total = pd.concat(total_scores)

    return df_metrics_total, df_scores_total
