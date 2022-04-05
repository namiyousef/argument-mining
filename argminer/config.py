import os
import pandas as pd

# -- get email details
EMAIL = os.environ.get('EMAIL', 'nlp.fyp1800@gmail.com')
EMAIL_PASSWORD = os.environ.get('EMAIL_PASSWORD', 'password')
EMAIL_RECIPIENTS = os.environ.get('EMAIL_RECIPIENTS', EMAIL)

# -- argument mining
PREDICTION_STRING_START_ID = 0

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
            'label_id': new_labels,
            'label': list(range(len(new_labels)))
        })
