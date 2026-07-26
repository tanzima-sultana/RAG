
import boto3
from config import S3_BUCKET

s3_client = boto3.client("s3")

def s3_key_exists(bucket, key):
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except s3_client.exceptions.ClientError:
        return False

def get_s3_key(s3_path):
    # self.output_path looks like: s3://S3_BUCKET/{Sub Directory}.parquet
    # strip the "s3://bucket/" prefix to get just the key
    prefix = f"s3://{S3_BUCKET}/"
    if not s3_path.startswith(prefix):
        raise ValueError(f"output_path '{s3_path}' does not start with expected prefix '{prefix}'")
    return s3_path[len(prefix):]

def get_s3_bucket_key(s3_path):
    # self.output_path looks like: s3://S3_BUCKET/{Sub Directory}.parquet
    # strip the "s3://bucket/" prefix to get just the key
    prefix = f"s3://{S3_BUCKET}/"
    if not s3_path.startswith(prefix):
        raise ValueError(f"output_path '{s3_path}' does not start with expected prefix '{prefix}'")
    
    return S3_BUCKET, s3_path[len(prefix):]

def s3_file_exists(s3_path):

    s3_key = get_s3_key(s3_path)
    resp = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=s3_key, MaxKeys=1)
    return resp.get("KeyCount", 0) > 0

    #return s3_key_exists(S3_BUCKET, s3_key)
