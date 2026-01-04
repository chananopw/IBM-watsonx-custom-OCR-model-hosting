#### Do manually. You must put your custom model on COS first. You might get your custom model from HiggingFace.
# export AWS_ACCESS_KEY_ID="<COS access key>"
# export AWS_SECRET_ACCESS_KEY="<COS secret access key>"

# Command
# aws --endpoint-url https://s3.us-south.cloud-object-storage.appdomain.cloud s3 cp <path to model in your local> s3://<bucket name>/<path...> --recursive --follow-symlinks

from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai import Credentials
from dotenv import load_dotenv
import os
from ibm_watsonx_ai.helpers.connections import DataConnection, S3Location


load_dotenv()

# X.ai cred
WX_AI_API_KEY = os.getenv("WX_AI_API_KEY")
WX_AI_URL = os.getenv("WX_AI_URL")
WX_AI_SPACE_ID = os.getenv("WX_AI_SPACE_ID")
# COS cred
COS_API_KEY = os.getenv("COS_API_KEY")
COS_ENDPOINT = os.getenv("COS_ENDPOINT")
COS_API_KEY_DESC = os.getenv("COS_API_KEY_DESC")
COS_API_KEY_ID = os.getenv("COS_API_KEY_ID")
COS_API_KEY_NAME = os.getenv("COS_API_KEY_NAME")
COS_IAM_ROLE_CRN = os.getenv("COS_IAM_ROLE_CRN")
COS_SERVICEID_CRN = os.getenv("COS_SERVICEID_CRN")
COS_ACCESS_KEY_ID = os.getenv("COS_ACCESS_KEY_ID")
COS_SECRET_ACCESS_KEY = os.getenv("COS_SECRET_ACCESS_KEY")
COS_BUCKET_NAME = os.getenv("COS_BUCKET_NAME")
CLOUD_IAM_URL = os.getenv("CLOUD_IAM_URL")

# allocate project
credentials = Credentials(url=WX_AI_URL, api_key=WX_AI_API_KEY)
client = APIClient(credentials)

client.set.default_space(space_id=WX_AI_SPACE_ID)
# client.task_credentials.store()
# print(client.task_credentials.get_details())

# Get COS connection
connection_details = client.connections.create({
    client.connections.ConfigurationMetaNames.NAME: "Connection to COS",
    client.connections.ConfigurationMetaNames.DATASOURCE_TYPE: client.connections.get_datasource_type_id_by_name('bluemixcloudobjectstorage'),
    client.connections.ConfigurationMetaNames.PROPERTIES: {
        'bucket': COS_BUCKET_NAME,
        'access_key': COS_ACCESS_KEY_ID,
        'secret_key': COS_SECRET_ACCESS_KEY,
        'iam_url': CLOUD_IAM_URL,
        'url': COS_ENDPOINT
    }
})

cos_connection_id = client.connections.get_uid(connection_details)
print(cos_connection_id)

# Get foundation model location from COS (Place your downloaded model in COS first)
sw_spec_id = client.software_specifications.get_id_by_name('watsonx-cfm-caikit-1.1')
metadata = {
    client.repository.ModelMetaNames.NAME: "typhoon_ocr_7b",  #change here
    client.repository.ModelMetaNames.SOFTWARE_SPEC_ID: sw_spec_id,
    client.repository.ModelMetaNames.TYPE: client.repository.ModelAssetTypes.CUSTOM_FOUNDATION_MODEL_1_0,
    client.repository.ModelMetaNames.MODEL_LOCATION: {
        "file_path": "model_storage", #change here
        "bucket": COS_BUCKET_NAME,
        "connection_id": cos_connection_id,
    },
}

stored_model_details = client.repository.store_model(model='scb10x/typhoon-ocr-7b', meta_props=metadata) #change here
# print(stored_model_details)
stored_model_asset_id = client.repository.get_model_id(stored_model_details)
print(client.repository.list())
print(stored_model_asset_id)

# Deploy model
meta_props = {
    client.deployments.ConfigurationMetaNames.NAME: "typhoon_ocr_7b_v7", #change here
    client.deployments.ConfigurationMetaNames.DESCRIPTION: "Testing deployment of embedding model",
    client.deployments.ConfigurationMetaNames.ONLINE: {},
    client.deployments.ConfigurationMetaNames.HARDWARE_SPEC: {
        "name": "1h100-80g", #change here 1h100-80g / 1a100-80g should be enough for 7b model
        "num_nodes": 1
    },
    # optionally overwrite model parameters here
    # client.deployments.ConfigurationMetaNames.FOUNDATION_MODEL: {"max_input_tokens": 256}, #change here or remove
    client.deployments.ConfigurationMetaNames.SERVING_NAME: "typhoon_ocr_7b_v7" #change here
}
deployment_details = client.deployments.create(stored_model_asset_id, meta_props)

deployment_id = client.deployments.get_id(deployment_details)
print(client.deployments.list())