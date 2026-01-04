import ibm_boto3
from ibm_botocore.client import Config, ClientError
import os
from dotenv import load_dotenv

load_dotenv()

# Constants
COS_ENDPOINT = "https://s3.jp-tok.cloud-object-storage.appdomain.cloud" # using what was in your script
COS_API_KEY_ID = "20h0jINDFF5SLGWtzKzcdgA_-5H2m94ne52ir1p3y7m_"
COS_INSTANCE_CRN = "crn:v1:bluemix:public:cloud-object-storage:global:a/8c9032b322ee42c8b18ea076b1af4fcb:428ccf5a-dde9-437a-b1fb-551d55614347:bucket:cloud-object-storage-cos-standard-fgi"
BUCKET_NAME = "cloud-object-storage-cos-standard-fgi"

print(f"Checking bucket: {BUCKET_NAME}")

if not COS_API_KEY_ID:
    print("Error: COS API Key missing from environment.")
    exit(1)

# Create resource
try:
    cos = ibm_boto3.resource("s3",
        ibm_api_key_id=COS_API_KEY_ID,
        ibm_service_instance_id=COS_INSTANCE_CRN,
        config=Config(signature_version="oauth"),
        endpoint_url=COS_ENDPOINT
    )
    
    bucket = cos.Bucket(BUCKET_NAME)
    print("Listing files in bucket:")
    count = 0
    for obj in bucket.objects.all():
        print(f" - {obj.key} ({obj.size} bytes)")
        count += 1
    
    if count == 0:
        print("Bucket is empty or access denied.")
    else:
        print(f"Found {count} objects.")

except Exception as e:
    print(f"Error accessing COS: {e}")
