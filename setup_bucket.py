from google.cloud import storage
from google.oauth2 import service_account
import json

KEY_PATH = "/mnt/c/Users/LABTI/Deforest.id/config/form-sembako-sa-key.json"
PROJECT = "form-sembako-chain"
BUCKET = "deforest-export"

credentials = service_account.Credentials.from_service_account_file(KEY_PATH)
client = storage.Client(project=PROJECT, credentials=credentials)

try:
    bucket = client.create_bucket(BUCKET, location="ASIA-SOUTHEAST2")
    print(f"Bucket created: {bucket.name}")
except Exception as e:
    print(f"Create bucket: {e}")

# Grant objectAdmin to the service account itself
# The service account already has permissions through its own credentials
# but we need to ensure the bucket policy allows it

bucket = client.get_bucket(BUCKET)
policy = bucket.get_iam_policy(requested_policy_version=3)
print(f"Bucket {bucket.name} exists, policy version: {policy.version}")
print("Ready for export!")
