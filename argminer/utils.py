from argminer.config import EMAIL, EMAIL_PASSWORD, EMAIL_RECIPIENTS
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import base64
import torch

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

