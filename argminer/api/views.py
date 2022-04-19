from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch
from argminer.data import ArgumentMiningDataset, PersuadeProcessor, TUDarmstadtProcessor
from argminer.api.utils import _generate_df_text_from_input
import pandas as pd
from argminer.config import  MODEL_MAP_DICT, LABELS_MAP_DICT
from argminer.utils import _get_label_maps

from argminer.evaluation import inference, _get_scores_agg
from torch.utils.data import DataLoader
def health_check():
    return 'OK'


# parameters to get
# dataset name, this will be fixed (enum)
# labelling strategy, this will be fixed (enum)
# aggregation strategy, this will be fixed (enum)
# model (prioritise our models, but if not then can create a new model and infer)
# max length
# input data
# batch_size
def evaluate(body, model_name, strategy, agg_strategy, strategy_level, max_length, batch_size, label_map):



    df_text = _generate_df_text_from_input(body, strategy)

    df_label_map = _get_label_maps(label_map, strategy)
    num_labels = df_label_map.shape[0]

    # num_classes after generating the df_label_map

    # -- LOAD MODEL AND TOKENIZER
    if model_name in MODEL_MAP_DICT:
        num_labels_gt = MODEL_MAP_DICT[model_name]['num_labels']
        if num_labels != num_labels_gt:
            return dict(
                error='The number of unique classes does not match that required by the model you requested',
                expected=num_labels_gt,
                received=f'{num_labels}: {df_label_map.label.values}'
            ), 400

        try:
            model = AutoModelForTokenClassification.from_pretrained(model_name)
            tokenizer_name = MODEL_MAP_DICT[model_name]['hugging_face_model_name']
        except:
            return dict(
                typr='model',
                name=model_name,
                error='Could not get item from HuggingFace'
            ),
            404
    else:
        try:
            model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
            tokenizer_name = model_name
        except Exception as e:
            return dict(
                typr='model',
                name=model_name,
                error='Could not get item from HuggingFace'
            ),
            404

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, add_prefix_space=True)
    except:
        return dict(
            typr='tokenizer',
            name=tokenizer_name,
            error='Could not get item from HuggingFace'
        ),
        404


    dataset = ArgumentMiningDataset(
        df_label_map,
        df_text,
        tokenizer,
        max_length,
        f'{strategy_level}_{strategy}',
        is_train=False
    )
    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size)

    df_metrics, df_scores, target_labels_full = inference(model, dataloader, return_labels=True)
    scores, df_scores_agg = _get_scores_agg(df_scores)
    df_scores_agg = df_scores_agg.reset_index()
    label_map = ['Other'] + [label for label in label_map if label != 'Other']
    df_scores_agg['class'] = df_scores_agg['class'].apply(lambda x: label_map[x])
    #target_labels_full = [[label_map[i] for i in target_label] for target_label in target_labels_full]

    return {'score_table': df_scores_agg.to_string().split('\n'), 'metrics':scores}, 200

def predict(body, model_name):

    dataset_name = MODEL_MAP_DICT[model_name].get('dataset')
    max_length = MODEL_MAP_DICT[model_name].get('max_length')
    strategy = model_name.split('_')[-1]
    df_label_map = LABELS_MAP_DICT[dataset_name][strategy]
    df_text = _generate_df_text_from_input([[f'Claim::{body}']], strategy)


    model = AutoModelForTokenClassification.from_pretrained(model_name)
    tokenizer_name = MODEL_MAP_DICT[model_name]['hugging_face_model_name']

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, add_prefix_space=True)



    fake_dataset = ArgumentMiningDataset(df_label_map, df_text, tokenizer, max_length, f'standard_{strategy}', is_train=False)
    fake_loader = DataLoader(fake_dataset)
    df_metrics, df_scores, target_labels_full = inference(model, fake_loader, return_labels=True)

    unique_labels = set()
    labels = []
    for label in df_label_map.label.values:
        if label != 'O':
            label = label.split('-')[-1]
        if label not in unique_labels:
            unique_labels.add(label)
            labels.append(label)
    df_return = pd.DataFrame().from_records(
        [(word, labels[label.item()]) for word, label in zip(body.split(), target_labels_full[0])],
        columns=['word', 'label']
    )
    return df_return.to_string().split('\n'), 200


def model_info(model_name):
    dataset_name = MODEL_MAP_DICT[model_name].get('dataset')
    strategy = model_name.split('_')[-1]

    hugging_face_model_name = MODEL_MAP_DICT[model_name].get('hugging_face_model_name')
    df_label_map = LABELS_MAP_DICT[dataset_name][strategy]

    return dict(
        hugging_face_model_name=hugging_face_model_name,
        labels=df_label_map.to_string().split('\n')
    ),
    200