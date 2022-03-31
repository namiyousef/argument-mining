from argminer.config import EMAIL, EMAIL_PASSWORD, EMAIL_RECIPIENTS
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import base64
import pandas as pd



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


    subject = f'JOB {job_name} COMPLETE'
    mail_content = '''
    JOB COMPLETION REPORT
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
    start_id = 1
    prev_doc = df.iloc[0].doc_id
    for (label, text, doc_id) in df[['label', 'text', 'doc_id']].itertuples(index=False):
        if doc_id != prev_doc:
            prev_doc = doc_id
            start_id = 1
        text_split = text.split()
        end_id = start_id + len(text_split)
        prediction_strings.append(
            [num for num in range(start_id, end_id)]
        )
        start_id = end_id
    df['predictionString'] = prediction_strings
    return df