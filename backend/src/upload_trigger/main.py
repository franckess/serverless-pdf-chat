import os, json
from datetime import datetime
import boto3
import PyPDF2
import shortuuid
import urllib
from aws_lambda_powertools import Logger

DOCUMENT_TABLE = os.environ["DOCUMENT_TABLE"]
MEMORY_TABLE = os.environ["MEMORY_TABLE"]
BUCKET = os.environ["BUCKET"]
KNOWLEDGE_BASE_DETAILS_SSM_PATH = os.environ["KNOWLEDGE_BASE_DETAILS_SSM_PATH"]


ddb = boto3.resource("dynamodb")
bedrock = boto3.client("bedrock")
document_table = ddb.Table(DOCUMENT_TABLE)
memory_table = ddb.Table(MEMORY_TABLE)
ssm = boto3.client("ssm")

s3 = boto3.client("s3")
logger = Logger()

def fix_json(json_string):
    # Replace single quotes with double quotes and escaping internal quotes if needed
    fixed_json = json_string.replace("'", '"')
    try:
        # Try loading the JSON to check if it's valid
        data = json.loads(fixed_json)
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON: {e}")
        return None

@logger.inject_lambda_context(log_event=True)
def lambda_handler(event, context):
    key = urllib.parse.unquote_plus(event["Records"][0]["s3"]["object"]["key"])
    split = key.split("/")
    user_id = split[0]
    file_name = split[1]

    document_id = shortuuid.uuid()

    s3.download_file(BUCKET, user_id, f"/tmp/{file_name}")

    with open(f"/tmp/{file_name}", "rb") as f:
        reader = PyPDF2.PdfReader(f)
        pages = str(len(reader.pages))

    conversation_id = shortuuid.uuid()

    timestamp = datetime.utcnow()
    timestamp_str = timestamp.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    document = {
        "userid": user_id,
        "documentid": document_id,
        "filename": file_name,
        "created": timestamp_str,
        "pages": pages,
        "filesize": str(event["Records"][0]["s3"]["object"]["size"]),
        "docstatus": "UPLOADED",
        "conversations": [],
    }

    conversation = {"conversationid": conversation_id, "created": timestamp_str}
    document["conversations"].append(conversation)

    document_table.put_item(Item=document)

    conversation = {"SessionId": conversation_id, "History": []}
    memory_table.put_item(Item=conversation)

    param_response = ssm.get_parameter(Name=KNOWLEDGE_BASE_DETAILS_SSM_PATH)
    print('***********************************')
    pprint.pp(param_response['Parameter']['Value'], depth=1)

    # Use the fix_json function to attempt to correct and parse the JSON
    knowledge_base_details = fix_json(param_response['Parameter']['Value'])
    if knowledge_base_details:
        try:
            bedrock.start_ingestion_job(
                knowledgeBaseId=knowledge_base_details['knowledgeBaseId'],
                dataSourceId=knowledge_base_details['dataSourceId']
            )
        except Exception as e:
            logger.error(f'Error triggering bedrock knowledge base sync: {e}')
    else:
        logger.error("Failed to fix and decode JSON from SSM parameter.")

    # Create metadata file
    metadata = {
        "metadataAttributes": {
            "userid": user_id,
            "documentid": document_id,
            "filename": file_name,
            "created": timestamp_str,
            "pages": pages
        }
    }
    metadata_file_path = f"/tmp/{file_name}.metadata.json"
    with open(metadata_file_path, 'w') as metadata_file:
        json.dump(metadata, metadata_file)

    # Upload metadata file to S3
    s3.upload_file(metadata_file_path, BUCKET, user_id)
