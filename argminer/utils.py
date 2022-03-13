from argminer.config import EMAIL, EMAIL_PASSWORD, EMAIL_RECIPIENTS
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def kaggle_split():
    """Function to split discourse text as done by Kaggle. See https://www.kaggle.com/c/feedback-prize-2021/discussion/297591 for full details
    :return:
    """

def send_job_completion_report(job_name):

    SENDER = EMAIL
    SENDER_PASSWORD = EMAIL_PASSWORD
    print(EMAIL_PASSWORD)
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