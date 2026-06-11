from google.oauth2 import service_account
from googleapiclient.discovery import build
import json

SCOPES = ["https://www.googleapis.com/auth/drive"]
KEY_PATH = "/mnt/c/Users/LABTI/Deforest.id/config/form-sembako-sa-key.json"

credentials = service_account.Credentials.from_service_account_file(KEY_PATH, scopes=SCOPES)
service = build("drive", "v3", credentials=credentials)

# Try to create a test folder in the service account's Drive
metadata = {
    "name": "deforest_test",
    "mimeType": "application/vnd.google-apps.folder"
}
try:
    folder = service.files().create(body=metadata, fields="id").execute()
    print(f"Test folder created: {folder.get('id')}")
    # Delete it
    service.files().delete(fileId=folder.get('id')).execute()
    print("Test folder deleted. Drive is working!")
except Exception as e:
    print(f"Drive error: {e}")
