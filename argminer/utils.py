# -- public imports
import os
import base64

import pandas as pd

import smtplib

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# -- private imports

# -- dev imports
from argminer.config import EMAIL, EMAIL_PASSWORD, EMAIL_RECIPIENTS, PREDICTION_STRING_START_ID, LABELS_MAP_DICT



def _get_label_maps(unique_labels, strategy):
    unique_labels = [label for label in unique_labels if label != 'Other']
    labels = ['O']
    if strategy == 'bio':
        for label in unique_labels:
            labels.append(f'B-{label}')
            labels.append(f'I-{label}')
    elif strategy == 'bieo':
        for label in unique_labels:
            labels.append(f'B-{label}')
            labels.append(f'I-{label}')
            labels.append(f'E-{label}')
    elif strategy == 'bixo':
        labels.append('X')
        for label in unique_labels:
            labels.append(f'B-{label}')
            labels.append(f'I-{label}')
    else:
        raise NotImplementedError(f'Strategy {strategy} has not implementation yet.')

    return pd.DataFrame({
        'label': labels
    }).reset_index().rename(columns={'index': 'label_id'})


# TODO these need to move to other packages as well
def send_job_completion_report(job_name):

    SENDER = EMAIL
    SENDER_PASSWORD = EMAIL_PASSWORD
    RECIEVER = EMAIL_RECIPIENTS

    # email parameters
    JOB_STATUS = 'SUCCESS' if os.path.exists('scores.json') else 'FAILED'
    MODEL_NAME = os.environ.get('MODEL_NAME')
    MAX_LENGTH = os.environ.get('MAX_LENGTH')
    EPOCHS = os.environ.get('EPOCHS')
    BATCH_SIZE = os.environ.get('BATCH_SIZE')
    VERBOSE = os.environ.get('VERBOSE')
    SAVE_FREQ = os.environ.get('SAVE_FREQ')
    TEST_SIZE = os.environ.get('TEST_SIZE')
    DATASET = os.environ.get('DATASET')
    STRATEGY = os.environ.get('STRATEGY')
    RUN_INFERENCE = os.environ.get('RUN_INFERENCE')

    subject = f'JOB {job_name}: {JOB_STATUS}'

    mail_content = f'''
    JOB PARAMETERS:
    ---------------
    MODEL_NAME: {MODEL_NAME}
    MAX_LENGTH: {MAX_LENGTH}

    DATASET: {DATASET}
    STRATEGY: {STRATEGY}
    TEST_SIZE: {TEST_SIZE}

    EPOCHS: {EPOCHS}
    BATCH_SIZE: {BATCH_SIZE}
    VERBOSE: {VERBOSE}
    SAVE_FREQ: {SAVE_FREQ}
    '''
    if RUN_INFERENCE:
        strat_level, strat_label = STRATEGY.split('_')
        df_label_map = LABELS_MAP_DICT[DATASET][strat_label]
        df_scores = pd.read_json('scores.json')
        df_label_map = df_label_map.merge(
            df_scores.groupby('class').mean().reset_index(), how='left', left_on='label_id', right_on='class'
        ).set_index('label')[['f1']]

        macro_f1 = df_label_map['f1'].mean()
        macro_f1_nan = df_label_map['f1'].fillna(0).mean()
        mail_content += f'''
    INFERENCE RESULTS:
    ------------------
    macro_f1: {macro_f1}
    macro_f1 with nan: {macro_f1_nan}

    DETAILED RESULTS:
    -----------------
    {df_label_map.to_string()}
    '''

    # Setup the MIME
    message = MIMEMultipart()
    message['From'] = SENDER
    message['To'] = RECIEVER
    message['Subject'] = subject

    # The body and the attachments for the mail
    message.attach(MIMEText(mail_content, 'plain'))

    # Create SMTP session for sending the mail
    session = smtplib.SMTP('smtp.gmail.com', 587)  # use gmail with port
    session.starttls()  # enable security
    session.login(SENDER, SENDER_PASSWORD)  # login with mail_id and password
    text = message.as_string()
    session.sendmail(SENDER, RECIEVER, text)
    session.quit()
    print('Mail successfully sent.')

def encode_model_name(model_name, epoch):
    model_name = f'{model_name}_{epoch}'
    model_name_b = model_name.encode('ascii')
    encoded_model_name = base64.b64encode(model_name_b).decode('ascii')
    return encoded_model_name

def decode_model_name(encoded_model_name):
    encoded_model_name_b = encoded_model_name.encode('ascii')
    model_name = base64.b64decode(encoded_model_name_b).decode('ascii')
    return model_name

def get_predStr(df):
    # TODO may not need, see about changes!
    assert all(item in list(df) for item in ['label', 'text', 'doc_id']), "Please use a dataframe with correct columns"
    prediction_strings = []
    start_id = PREDICTION_STRING_START_ID
    prev_doc = df.iloc[0].doc_id
    for (label, text, doc_id) in df[['label', 'text', 'doc_id']].itertuples(index=False):
        if doc_id != prev_doc:
            prev_doc = doc_id
            start_id = PREDICTION_STRING_START_ID
        text_split = text.split()
        end_id = start_id + len(text_split)
        prediction_strings.append(
            [num for num in range(start_id, end_id)]
        )
        start_id = end_id
    df['predictionString'] = prediction_strings
    return df