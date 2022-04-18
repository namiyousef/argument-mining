# -- public imports
import io
import os
import json
import base64

import pandas as pd

import smtplib

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

import matplotlib.pyplot as plt
# -- private imports

# -- dev imports
from argminer.config import EMAIL, EMAIL_PASSWORD, EMAIL_RECIPIENTS, PREDICTION_STRING_START_ID, LABELS_MAP_DICT



def _get_label_maps(unique_labels, strategy):
    unique_labels = [label for label in unique_labels if label != 'Other']
    labels = ['O']
    if strategy == 'io':
        for label in unique_labels:
            labels.append(f'I-{label}')
    elif strategy == 'bio':
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
    if RUN_INFERENCE and JOB_STATUS == 'SUCCESS':
        strat_level, strat_label = STRATEGY.split('_')
        df_label_map = LABELS_MAP_DICT[DATASET][strat_label]
        df_scores = pd.read_json('scores.json')
        df_label_map = df_label_map.merge(
            df_scores.groupby('class').mean().reset_index(), how='left', left_on='label_id', right_on='class'
            # TODO this will fail hard if you make changes to the code. Needs to be more robust for reporting
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

    with open('training_scores.json', 'r') as f:
        scores_dict = json.load(f)

    epoch_f_scores = scores_dict['epoch_scores']['FScore']
    epoch_batch_ids = scores_dict['epoch_batch_ids']['FScore']
    f_scores = scores_dict['scores']['FScore']

    batches = range(len(f_scores))
    plt.plot(batches, f_scores)
    plt.plot(epoch_batch_ids, epoch_f_scores, 'or')


    plt.xlabel('Time')
    plt.ylabel('FScore')

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    img_data = buffer.read()


    # Setup the MIME
    message = MIMEMultipart()
    message['From'] = SENDER
    message['To'] = RECIEVER
    message['Subject'] = subject

    # The body and the attachments for the mail
    message.attach(MIMEText(mail_content, 'plain'))
    message.attach(MIMEImage(img_data))

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
    assert all(item in list(df) for item in ['label', 'text', 'doc_id']), "Please use a dataframe with correct columns"
    prediction_strings = []
    start_id = PREDICTION_STRING_START_ID
    prev_doc = df.iloc[0].doc_id
    for (text, doc_id) in df[['text', 'doc_id']].itertuples(index=False):
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


def _generate_job_scripts(DATASET, MODEL_NAME, EPOCHS, STRATEGY):
    assert DATASET in {'Persuade', 'TUDarmstadt'}
    assert MODEL_NAME in {'google/bigbird-roberta-base', 'roberta-base'}
    assert STRATEGY in {'io', 'bio', 'bieo'}
    # inferred variables

    if DATASET == 'Persuade' and EPOCHS == 20:
        HOURS = 24
    else:
        HOURS = 6

    if MODEL_NAME == 'google/bigbird-roberta-base':
        MAX_LENGTH = 1024
        MODEL_SHORT = 'bigbird'
    else:
        MAX_LENGTH = 512
        MODEL_SHORT = 'roberta'

    JOB_NAME = f'{DATASET}_e{EPOCHS}_{STRATEGY}_{MODEL_SHORT}'

    script_text = f'''#!/bin/bash -l
#$ -m be

# Request ten minutes of wallclock time (format hours:minutes:seconds).
#$ -l h_rt={HOURS}:00:0

# request GPU node
#$ -l gpu=1

# request A100 GPU node
# #$ -ac allow=L

# Request RAM (must be an integer followed by M, G, or T)
#$ -l mem=32G

# Request temp space
#$ -l tmpfs=15G

# Set the name of the job.
#$ -N {JOB_NAME}

nvidia-smi

# ENV VARIABLES
# -- email configs
export EMAIL_PASSWORD="NLP.FYP1800"
export EMAIL_RECIPIENTS="ucabyn0@ucl.ac.uk, ucabfd0@ucl.ac.uk, ucabqfe@ucl.ac.uk, ucabc21@ucl.ac.uk, qingyu.feng.21@ucl.ac.uk, changmao.huang.21@ucl.ac.uk"

# -- model specific configs
export MODEL_NAME="{MODEL_NAME}"
export MAX_LENGTH={MAX_LENGTH}

# -- training configs
export EPOCHS={EPOCHS}
export BATCH_SIZE=4
export VERBOSE=2
export SAVE_FREQ=10
export TEST_SIZE="0.3"

# -- dataset configs
export DATASET="Persuade"

# -- experiment configs
export STRATEGY_LEVEL="standard"
export STRATEGY_NAME="{STRATEGY}"
export STRATEGY="${{STRATEGY_LEVEL}}_${{STRATEGY_NAME}}"
export RUN_INFERENCE=1

# -- inferred variables
export JSON_FILE_NAME="${{DATASET}}_postprocessed.json"
export DATA_PATH="data/${{STRATEGY_NAME}}/${{JSON_FILE_NAME}}"

# TODO this is a hotfix due to darmstadt processor tts. Needs cleaning
export TTS_FILE="train-test-split.csv"
export TTS_PATH="data/${{TTS_FILE}}"
cp -r $TTS_PATH $TMPDIR/$TTS_FILE

# Set the working directory to somewhere in your scratch space.
#  This is a necessary step as compute nodes cannot write to $HOME.
# Replace "<your_UCL_id>" with your UCL user ID.
#$ -wd /home/ucabyn0/Scratch

# COPY NECESSARY FILES
cp -r job_files/run.py $TMPDIR/run.py
cp -r $DATA_PATH $TMPDIR/$JSON_FILE_NAME
cp -r venv $TMPDIR/venv

cd $TMPDIR


# LOAD MODULES
module unload compilers mpi
module load compilers/gnu/4.9.2
module load python/3.7.4
module load cuda/10.1.243/gnu-4.9.2
module load cudnn/7.5.0.56/cuda-10.1

# venv should have the most recent version of argminer installed
source venv/bin/activate



python3 -c "import torch; print(f'GPU Availability: torch.cuda.is_available()')"
python3 run.py $DATASET $STRATEGY $MODEL_NAME $MAX_LENGTH -test-size=$TEST_SIZE -b=$BATCH_SIZE -e=$EPOCHS -save-freq=$SAVE_FREQ -verbose=$VERBOSE -i=$RUN_INFERENCE
python3 -c "from argminer.utils import send_job_completion_report; send_job_completion_report('${{JOB_ID}}')"


# nvidia-smi

tar -zcvf $HOME/Scratch/files_from_job_$JOB_ID.tar.gz $TMPDIR

env
                '''
    with open(f'job_scripts/{JOB_NAME}.sh', 'w') as f:
        f.write(script_text)


#if __name__ == '__main__':
#    for dataset in {'Persuade', 'TUDarmstadt'}:
#        for model_name in {
#            #'google/bigbird-roberta-base',
#            'roberta-base'
#        }:
#            for strategy in {'io', 'bio', 'bieo'}:
#                for epochs in {5, 20}:
#                    _generate_job_scripts(
#                        dataset, model_name, epochs, strategy
#                    )
