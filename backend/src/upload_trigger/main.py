import os, json
from datetime import datetime
import boto3
import PyPDF2
import shortuuid
import urllib
from aws_lambda_powertools import Logger
import pprint
import requests  # You might need to install this with pip if it's not already available

DOCUMENT_TABLE = os.environ["DOCUMENT_TABLE"]
MEMORY_TABLE = os.environ["MEMORY_TABLE"]
BUCKET = os.environ["BUCKET"]
KNOWLEDGE_BASE_DETAILS_SSM_PATH = os.environ["KNOWLEDGE_BASE_DETAILS_SSM_PATH"]


ddb = boto3.resource("dynamodb")
bedrock = boto3.client("bedrock-agent")
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

def set_doc_status(user_id, document_id, status):
    document_table.update_item(
        Key={"userid": user_id, "documentid": document_id},
        UpdateExpression="SET docstatus = :docstatus",
        ExpressionAttributeValues={":docstatus": status},
    )

@logger.inject_lambda_context(log_event=True)
def lambda_handler(event, context):
    key = urllib.parse.unquote_plus(event["Records"][0]["s3"]["object"]["key"])
    split = key.split("/")
    user_id = split[0]
    file_name = split[1]

    document_id = shortuuid.uuid()
    s3.download_file(BUCKET, key, f"/tmp/{file_name}")

    # Generate a presigned URL for uploading a file using PUT operation
    presigned_url = s3.generate_presigned_url(
        ClientMethod='put_object',
        Params={
            'Bucket': BUCKET,
            'Key': f"{user_id}/{file_name}",  # Directly using user_id and file_name
            'ContentType': 'application/octet-stream'  # Adjust ContentType based on your file type
        },
        ExpiresIn=600,   # URL expires in 10 minutes
        HttpMethod="PUT",
    )

    # Log the presigned URL (for debugging purposes, remove in production)
    logger.info(f"Generated presigned URL: {presigned_url}")

    # Assuming the file is already in the right place or handled by the client using the presigned URL
    # No need to download the file, as the client will upload it directly using the presigned URL

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

    set_doc_status(user_id, document_id, "READY")
    
    # Generate metadata file
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

    # Generate a presigned URL for uploading the metadata file
    presigned_url = s3.generate_presigned_url(
        ClientMethod='put_object',
        Params={
            'Bucket': BUCKET,
            'Key': f"{user_id}/{file_name}.metadata.json",
            'ContentType': 'application/json'  # Set appropriate content type for JSON
        },
        ExpiresIn=600  # URL expires in 10 minutes
    )

    # Upload the metadata file using the presigned URL
    with open(metadata_file_path, 'rb') as metadata_file:
        files = {'file': metadata_file}
        response = requests.put(presigned_url, data=metadata_file.read(), headers={'Content-Type': 'application/json'})

    if response.status_code == 200:
        logger.info("Metadata file uploaded successfully using presigned URL.")
    else:
        logger.error(f"Failed to upload metadata file: {response.text}")
