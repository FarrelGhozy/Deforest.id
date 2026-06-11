from googleapiclient import discovery
from google.oauth2 import service_account
import json

KEY_PATH = "/mnt/c/Users/LABTI/Deforest.id/config/form-sembako-sa-key.json"
PROJECT = "form-sembako-chain"

credentials = service_account.Credentials.from_service_account_file(
    KEY_PATH,
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)

# Enable Drive API
serviceusage = discovery.build("serviceusage", "v1", credentials=credentials)
name = f"projects/{PROJECT}/services/drive.googleapis.com"

try:
    result = serviceusage.services().enable(name=name).execute()
    print("Drive API enabled:", result)
except Exception as e:
    print("Enable Drive API:", e)

# Check Drive API status
try:
    result = serviceusage.services().get(name=name).execute()
    print(f"Drive API state: {result.get('state')}")
except Exception as e:
    print("Check Drive API:", e)
