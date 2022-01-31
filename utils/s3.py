import logging
import os

import boto3
import pandas as pd
from botocore.exceptions import ClientError

s3 = boto3.client('s3', aws_access_key_id='AKIA25ZCC7LDUWZTJEMQ',
                  aws_secret_access_key='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')


def load_file(file: str, type='json'):
    obj = s3.get_object(Bucket='tweetstfm', Key=file)
    if type == 'json':
        tweets = pd.read_json(obj.get("Body"))
        return tweets
    if type == 'csv':
        tweets = pd.read_csv(obj.get("Body"))
        return tweets


def upload_file(file_name, bucket='tweetstfm', object_name=None):
    if object_name is None:
        object_name = os.path.basename(file_name)
    try:
        response = s3.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True
