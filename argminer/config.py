import os
import pandas as pd

# -- get email details
EMAIL = os.environ.get('EMAIL', 'nlp.fyp1800@gmail.com')
EMAIL_PASSWORD = os.environ.get('EMAIL_PASSWORD', 'password')
EMAIL_RECIPIENTS = os.environ.get('EMAIL_RECIPIENTS', EMAIL)

# -- argument mining
PREDICTION_STRING_START_ID = 0
MAX_NORM = 10
# -- label maps
# TODO automate these...
LABELS_DICT = dict(
    TUDarmstadt=['MajorClaim', 'Claim', 'Premise'],
    Persuade=['Lead', 'Position', 'Claim', 'Counterclaim', 'Rebuttal', 'Evidence', 'Concluding Statement']
)
STRATEGIES = ['io', 'bio', 'bieo', 'bixo']

LABELS_MAP_DICT = {}
for dataset, labels in LABELS_DICT.items():
    LABELS_MAP_DICT[dataset] = {}
    for strategy in STRATEGIES:
        new_labels = ['O']
        if strategy == 'bixo':
            new_labels.append('X')
        for label in labels:
            if 'b' in strategy:
                new_labels.append(f'B-{label}')
            new_labels.append(f'I-{label}')
            if 'e' in strategy:
                new_labels.append(f'E-{label}')
        LABELS_MAP_DICT[dataset][strategy] = pd.DataFrame({
            'label_id': list(range(len(new_labels))),
            'label': new_labels
        })


MODEL_MAP_DICT = {
    'ucabqfe/bigBird_PER_io': dict(
        dataset='Persuade',
        hugging_face_model_name="google/bigbird-roberta-base",
        max_length=1024,
        num_labels=8
    ),
    'ucabqfe/bigBird_PER_bio': dict(
        dataset='Persuade',
        hugging_face_model_name="google/bigbird-roberta-base",
        max_length=1024,
        num_labels=15
    ),
    'ucabqfe/bigBird_PER_bieo': dict(
        dataset='Persuade',
        hugging_face_model_name="google/bigbird-roberta-base",
        max_length=1024,
        num_labels=22
    ),
    'ucabqfe/bigBird_AAE_io': dict(
        dataset='TUDarmstadt',
        hugging_face_model_name="google/bigbird-roberta-base",

        num_labels=4
    ),
    'ucabqfe/bigBird_AAE_bio': dict(
        dataset='TUDarmstadt',
        hugging_face_model_name="google/bigbird-roberta-base",
        max_length=1024,
        num_labels=7
    ),
    'ucabqfe/bigBird_AAE_bieo': dict(
        dataset='TUDarmstadt',
        hugging_face_model_name="google/bigbird-roberta-base",
        max_length=1024,
        num_labels=10
    ),
    'ucabqfe/roberta_PER_io': dict(
        dataset='Persuade',
        hugging_face_model_name="roberta-base",
        max_length=512,
        num_labels=8
    ),
    'ucabqfe/roberta_PER_bio': dict(
        dataset='Persuade',
        hugging_face_model_name="roberta-base",
        max_length=512,
        num_labels=15
    ),
    'ucabqfe/roberta_PER_bieo': dict(
        dataset='Persuade',
        hugging_face_model_name="roberta-base",
        max_length=512,
        num_labels=22
    ),
    'ucabqfe/roberta_AAE_io': dict(
        dataset='TUDarmstadt',
        hugging_face_model_name="roberta-base",
        max_length=512,
        num_labels=4
    ),
    'ucabqfe/roberta_AAE_bio': dict(
        dataset='TUDarmstadt',
        hugging_face_model_name="roberta-base",

        num_labels=7
    ),
    'ucabqfe/roberta_AAE_bieo': dict(
        dataset='TUDarmstadt',
        hugging_face_model_name="roberta-base",
        max_length=512,
        num_labels=10
    ),

}