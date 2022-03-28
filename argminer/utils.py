from argminer.config import EMAIL, EMAIL_PASSWORD, EMAIL_RECIPIENTS
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import base64
import torch
import pandas as pd

# TODO this needs to be moved to a separate package. Perhaps torchutils?
def _first_appearance_of_unique_item(x):
    """ Torch function to get the first appearance of a unique item"""
    unique, inverse = torch.unique(x, sorted=True, return_inverse=True)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    perm = inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)
    return perm


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

def kaggle_split():
    """Function to split discourse text as done by Kaggle. See https://www.kaggle.com/c/feedback-prize-2021/discussion/297591 for full details
    :return:
    """

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

def _move(data, device='cpu'):
    if torch.is_tensor(data):
        return data.to(device)
    elif isinstance(data, dict):
        return {key: tensor.to(device) for key, tensor in data.items()}
    elif isinstance(data, list):
        raise NotImplementedError('Currently no support for tensors stored in lists.')
    else:
        raise TypeError('Invalid data type.')

