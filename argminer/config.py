import os

# -- get email details
EMAIL = os.environ.get('EMAIL', 'nlp.fyp1800@gmail.com')
EMAIL_PASSWORD = os.environ.get('EMAIL_PASSWORD', 'password')
EMAIL_RECIPIENTS = os.environ.get('EMAIL_RECIPIENTS', EMAIL)

# -- argument mining
PREDICTION_STRING_START_ID = 0
