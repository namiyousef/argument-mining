# -- public imports
import gc
import torch
import warnings
import pandas as pd
import time # TODO remove this after debug

# -- private imports
from colabtools.config import DEVICE
from colabtools.utils import move_to_device

# -- dev imports
from argminer.config import PREDICTION_STRING_START_ID

def get_word_labels(inputs, outputs, agg_strategy, has_x):
    """
    Function that aggregates from subtokens back to words given an aggregation strategy,
    probabilities for each subtoken, and word_ids. This can also be used for targets with the restriction
    that agg_strategy='first'

    :param inputs: word ids that map each subtoken to it's word. This comes from tokenizer(string).word_ids()
    but has the None parameters replaced with a value less than zero (e.g. -100) indicating that it is not a word (e.g. SEP, CLS)
    :type inputs: torch.Tensor
    :param outputs: raw predictions from a model output (OR the target variables)
    :type outputs: torch.Tensor (of dtype=torch.int64 for targets)
    :param agg_strategy: defines how the subtokens are aggregated back to words. This takes 'max', 'first' or 'mean'
    :type agg_strategy: str
    :param has_x: flag to indicates whether to ignore subtokens or not. Automatically defers to agg_strategy='first' if true
    :type has_x: bool

    :returns: list of shortened tensors corresponding to each word
    :rtype: list
    """

    # -- CONFIGURATION
    prediction_shape = outputs[0].shape
    if len(prediction_shape) > 1:
        feature_dim = prediction_shape[-1]
    else:
        feature_dim = 1
        if outputs.dtype == torch.int64 and agg_strategy != 'first':
            raise ValueError('agg_strategy must be "first" if aggregating targets vector.')

    if has_x and agg_strategy != 'first':
        warnings.warn(
            f'agg_strategy="{agg_strategy}" with has_x={has_x} is not compatible. '
            f'Instead aggregation with agg_strategy="first" will apply.', UserWarning, stacklevel=2
        )


    pred_labels = []
    for (predictions, word_ids) in zip(outputs, inputs):

        # filter items that don't correspond to words
        mask = word_ids >= 0
        word_ids = word_ids[mask]
        predictions = predictions[mask]

        unique_word_ids, word_id_counts = torch.unique_consecutive(word_ids, return_counts=True)
        agg_predictions = torch.zeros((unique_word_ids.shape[0], feature_dim), dtype=predictions.dtype)

        start_id = 0
        for i, (unique_word_id, word_id_count) in enumerate(zip(unique_word_ids, word_id_counts)):

            end_id = start_id + word_id_count

            # get segments corresponding to word
            prediction_slice = predictions[start_id: end_id]

            # apply aggregation strategy
            if agg_strategy == 'mean' and not has_x:
                agg_predictions[i] = prediction_slice.mean(dim=0)
            elif agg_strategy == 'max' and not has_x:
                agg_predictions[i], _ = prediction_slice.max(dim=0)
            else:
                agg_predictions[i] = prediction_slice[0]
            start_id = end_id

        pred_labels.append(agg_predictions)
        
    return pred_labels

def get_predictionString(labels, doc_ids):
    """
    Function that takes list of tensors along with unique document ids to generate a dataframe
    with predictionStrings for each (doc_id, class).

    :param labels: list of tensors that contain classes, must be of type torch.int64
    :type labels: list
    :param doc_ids: collection of document ids pertaining to each item in labels
    :type doc_ids: torch.Tensor

    :returns: dataframe in the following form (doc_id, class, predictionString) where predictionString is a set
    """

    ids = []
    classes = []
    prediction_strings = []

    for doc_id, label in zip(doc_ids, labels):
        unique_classes, unique_class_counts = torch.unique_consecutive(label, return_counts=True)
        # define prediction string start
        start_id = PREDICTION_STRING_START_ID
        for unique_class, unique_class_count in zip(unique_classes, unique_class_counts):
            ids.append(doc_id.item())
            classes.append(unique_class.item())
            end_id = start_id + unique_class_count + 1
            prediction_strings.append(set(range(start_id, end_id)))
            start_id = end_id
    return pd.DataFrame(data={'id': ids, 'class': classes, 'predictionString': prediction_strings})

def evaluate(df_outputs, df_targets, threshold=0.5):
    """
    Calculates the macro f1 score for a given batch of data. This Macro F1 score is based on the following:
    https://www.kaggle.com/competitions/feedback-prize-2021/overview/evaluation

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
    # TODO add thresholds here
    df_targets['tp'] = df_targets[[overlap_pred_gt, overlap_gt_pred]].apply(
        lambda x: int(x.overlap_pred_gt >= threshold and x.overlap_gt_pred >= threshold), axis=1
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
    # TODO double check F1 score correct
    scores = scores.assign(f1=lambda x: x['tp'] / (x['tp'] + 1 / 2 * (x['fp'] + x['fn'])))

    return scores

def inference(model, testloader, metrics=[], return_labels=False):
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

    target_labels_full = []

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(testloader):

            inputs = move_to_device(inputs, DEVICE)
            targets = move_to_device(targets, DEVICE)

            s = time.time() # TODO add verbose statement
            loss, outputs = model(
                labels=targets,
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                return_dict=False
            )
            print(f'Prediction time: {time.time() - s:.3g}')

            word_ids = inputs['word_ids']
            doc_ids = inputs['index']

            # measure performance at subtoken level
            df_metrics_no_agg = pd.DataFrame.from_records({
                f'{metric.__class__.__name__}_no_agg': metric(outputs, targets) for metric in metrics
            })

            s = time.time()
            # aggregate from subtoken to words # TODO strat needs to be parametrised
            word_label_probas = get_word_labels(word_ids, outputs, agg_strategy='first', has_x=False)
            word_label_ids = [tensor.argmax(dim=1) for tensor in word_label_probas]

            # aggregate from subtoken to words for targets. Note: agg_strategy always first
            target_label_probas = get_word_labels(word_ids, targets, agg_strategy='first', has_x=False)
            # TODO this is a hotfix. Need an automatic dimensioning tool!
            target_label_ids = [tensor.flatten() for tensor in target_label_probas]


            # TODO try on GPU and optimise memory and speed. Perform pandas operations on CPU
            del targets, outputs
            gc.collect()
            torch.cuda.empty_cache()


            # measure performance at word level, before labels are mapped to reduced form
            # TODO double check metrics here is working
            df_metrics_agg = pd.DataFrame.from_records({
                f'{metric.__class__.__name__}_agg': [
                    metric(output, target).item() for output, target in zip(word_label_ids, target_label_ids)
                ] for metric in metrics
            })

            # map word labels to reduced form
            word_labels = [reduce_map_values[label_ids] for label_ids in word_label_ids]
            target_labels = [reduce_map_values[label_ids] for label_ids in target_label_ids]

            if return_labels:
                for target_label in target_labels:
                    target_labels_full.append(target_label)
            print(f'Agg to word time: {time.time() - s:.3g}')


            # measure performance at word level, for reduced labels
            df_metrics_agg_reduced = pd.DataFrame.from_records({
                f'{metric.__class__.__name__}_agg_reduced': [
                    metric(output, target).item() for output, target in zip(word_labels, target_labels)
                ] for metric in metrics
            })

            # combine all metrics
            df_metrics = pd.concat([df_metrics_no_agg, df_metrics_agg, df_metrics_agg_reduced], axis=1)


            # get dataframes of (doc_id, class, prediction_string)
            s = time.time()
            df_targets_predString = get_predictionString(target_labels, doc_ids)
            df_outputs_predString = get_predictionString(word_labels, doc_ids)

            print(f'Get predstring time: {time.time() - s:.3g}')


            # get scores (and macro f1)
            s = time.time()
            df_scores = evaluate(df_outputs_predString, df_targets_predString)
            print(f'Evaluate time: {time.time() - s:.3g}')

            total_metrics.append(df_metrics)
            total_scores.append(df_scores)
            print(f'Batch {i+1} complete.')

        # TODO inference is very slow....
    df_metrics_total = pd.concat(total_metrics)
    df_scores_total = pd.concat(total_scores).reset_index(drop=True)
    # TODO add a reduce operation on inference
    if target_labels_full:
        return df_metrics_total, df_scores_total, target_labels_full
    else:
        return df_metrics_total, df_scores_total

def _get_scores_agg(df):
    df = df.groupby('class').sum()
    df['f1'] = df.tp / (df.tp + 1/2*(df.fp + df.fn))
    df['recall'] = df.tp / (df.tp + df.fn)
    df['precision'] = df.tp / (df.tp + df.fp)
    scores = {'macro_f1':df['f1'].mean(), 'macro_recall': df['recall'].mean(), 'macro_precision': df['precision'].mean()}
    return scores, df