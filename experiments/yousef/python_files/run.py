import torch
import time
from python_files.utils import _move # TODO move this to pytorch utils
from colabtools.utils import get_gpu_utilization
import pandas as pd



if torch.cuda.is_available():
    DEVICE = 'cuda'
    print('CUDA device detected. Using GPU...')
else:
    DEVICE = 'cpu'
    print('CUDA device NOT detected. Using CPU...')

def train_longformer(model, optimizer, epochs, train_loader, val_loader=None, verbose=2):
    """
    Function to train longformer
    :return:
    """
    model.to(DEVICE)
    model.train()

    for epoch in range(epochs):
        #print(f'GPU Utilisation at epoch {epoch}: {get_gpu_utilization()}')
        # set model to train mode
        model.train()
        start_epoch_message = f'EPOCH {epoch + 1} STARTED'
        print(start_epoch_message)
        print(f'{"-" * len(start_epoch_message)}')

        start_epoch = time.time()
        # TODO model does not currently support saving
        start_load = time.time()

        training_loss = 0
        for i, (inputs, targets) in enumerate(train_loader):
            start_train = time.time()
            inputs = _move(inputs)
            targets = _move(targets)
            #print(f'GPU Utilisation at batch {i+1} after data loading: {get_gpu_utilization()}')

            optimizer.zero_grad()

            loss, outputs = model(
                labels=targets,
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                return_dict=False
            )
            #print(f'GPU Utilisation at batch {i+1} after training: {get_gpu_utilization()}')

            print(loss)
            print(outputs, outputs.shape)
            training_loss += loss.item()

            """def active_logits(raw_logits, word_ids):
                word_ids = word_ids.view(-1)
                active_mask = word_ids.unsqueeze(1).expand(word_ids.shape[0], HyperParameters.num_labels)
                active_mask = active_mask != NON_LABEL
                active_logits = raw_logits.view(-1, HyperParameters.num_labels)
                active_logits = torch.masked_select(active_logits, active_mask)  # return 1dTensor
                active_logits = active_logits.view(-1, HyperParameters.num_labels)
                return active_logits"""

            # TODO evaluate code here

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            end_train = time.time()

            if verbose > 1:
                print(
                    f'Batch {i + 1} complete. Time taken: load({start_train - start_load:.3g}), '
                    f'train({end_train - start_train:.3g}), total({end_train - start_load:.3g}). '
                )
            start_load = time.time()


def _get_zero_start_ids(iterable, split_by=[0]):
    """ TODO run unit tests on this """

    split_by = set(split_by)
    add_item = float(iterable[0]) in split_by
    zero_start_ids = []
    for i, item in enumerate(iterable):
        if (float(item) in split_by) ^ add_item:
            add_item = not add_item
            zero_start_ids.append(i)
    return zero_start_ids

def chunk_into_contiguous_segments(tensor, split_ids, ignore=False):
    tensor_split = torch.tensor_split(tensor, split_ids)
    if ignore:
        tensor_split = [tensor_split for i, chunk in enumerate(tensor_split) if i%2]
    return tensor_split


def get_predicted_label(outputs):
    probas, labels = outputs.max(dim=-1)
    return probas, labels

def _get_predictionString_from_tensors(word_id_masks, word_ids, outputs, strategy='first'):
    """ Function to convert predicted token labels back into words """
    # outputs is of size batch_size, max_length, n_labels
    if strategy == 'first':
        # also you need to know what output to be expected here, if it's gonna be
        # diff strategies you might have to give the WHOLE tensor, not just the highest pred and class
        probas, labels = get_predicted_label(outputs)
        df = pd.concat([
            pd.DataFrame({
                'probas': proba[word_id_mask],
                'labels': label[word_id_mask],
                'word_ids': word_id,
                'doc_ids': i
            }).groupby('word_ids').head(1) for i, (proba, label, word_id, word_id_mask) in enumerate(zip(probas, labels, word_ids, word_id_masks))
        ])

        df = df.groupby('ids').agg({col: lambda x: x.values.flatten().tolist() for col in df.columns}).drop('ids', axis=1)
        # also you need to convert back top pytorch tensors
        df['split_ids'] = df.labels.apply(lambda x: _get_zero_start_ids(x))
        df.probas = df[['probas', 'split_ids']].apply(lambda x: c)
        # continue this...


    else:
        raise NotImplementedError(f'Strategy {strategy} has not yet been implemented.')
    # needs to return tensor/df of following format
    # TODO need multiple strategies for calculating the CLASS. This might depend on the tokenisation
    # DOC_ID, CLASS, predictionString (word level)
    # find location of first zero section
    zero_start_ids = [
        _get_zero_start_ids(arr) for arr in arrs
    ]
    pass

def _calculate_f1_score():
    pass

def test(model, test_loader, verbose=2):
    """
    Performs inference on a trained model
    :param model:
    :param test_loader:
    :param verbose:
    :return:
    """
    # TODO give option not to use GPU, in fact maybe you can't even
    model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        start_load = time.time()
        for i, (inputs, targets) in enumerate(test_loader):

            start_get_predString = time.time()

            start_calculate_f1 = time.time()

            end_prediction = time.time()
            if verbose > 1:
                print(
                    f'Batch {i + 1} complete. Time taken: '
                    f'load({start_get_predString - start_load:.3g}), '
                    f'calc_predString({start_calculate_f1 - start_get_predString:.3g}), '
                    f'calc_f1({end_prediction - start_calculate_f1}), '
                    f'total({end_prediction - start_load:.3g}). '
                )
            # needs to convert the inputs to word level
            # needs to calculate F1 score
            # needs to average across the whole thing
            pass

    return None


def evaluate_simple(outputs, targets):
    """ Function that evaluates model predictions at word level based on an F1 Metric
    This is a simple evaluation that checks a document level
    """
    ids_equal = outputs == targets
    overlap = (ids_equal.float()).mean(dim=1)
    ids_tp = overlap >= 0.5

    return (ids_equal.float()).mean(dim=1)


def _word_labels_to_df(passage_prediction_list, doc_ids):
    ids = []
    classes = []
    prediction_strings = []
    for doc_id, passage_prediction in zip(doc_ids, passage_prediction_list):
        pred_string_list = []
        for prediction_string, label in enumerate(passage_prediction, 1):
            if label != 0:
                pred_string_list.append(prediction_string)
            else:
                if pred_string_list:
                    ids.append(doc_id)
                    classes.append(label)
                    prediction_strings.append(set(pred_string_list))
                    pred_string_list = []

    return pd.DataFrame(data={'id': ids, 'class': classes, 'predictionString': prediction_strings})


def evaluate(output_passage_prediction, target_passage_prediction):

    df_outputs = _word_labels_to_df(output_passage_prediction)
    df_targets = _word_labels_to_df(target_passage_prediction)

    # assume you can get dataframes
    df_targets = df_targets.merge(
        df_outputs, on=['id', 'class']
    )

    # make sure class_y is the correct one! e.g. the ground truth
    df_targets['overlap_pred_gt'] = df_targets[['class_x', 'class_y']].apply(
        lambda x: x.class_x.intersection(x.class_y) / len(x.class_y)
    )
    df_targets['overlap_gt_pred'] = df_targets[['class_y', 'class_x']].apply(
        lambda x: x.class_x.intersection(x.class_y) / len(x.class_x)

    )

    df_targets['tp'] = df_targets[['overlap_pred_gt', 'overlap_gt_pred']].apply(
        lambda x: int(x.overlap_pred_gt >= 0.5 and x.overlap_gt_pred >= 0.5)
    )
    df_targets['overlap_sum'] = df_targets.overlap_pred_gt + df_targets.overlap_gt_pred
    df_targets_match = df_targets[df_targets['tp'] == 1].groupby(
        'class_y'
    ).agg({'overlap_sum':'max'}).drop(['overlap_gt_pred', 'overlap_pred_gt'])

    TP = df_targets.shape[0]

    df_targets_no_match = df_targets[df_targets['tp'] == 0]
    FN = len(df_targets_no_match.class_y.unique())
    FP = len(df_targets_no_match.class_x.unique())
    pass


