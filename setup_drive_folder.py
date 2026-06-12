from google.oauth2 import service_account
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/drive"]
KEY_PATH = "/mnt/c/Users/LABTI/Deforest.id/config/form-sembako-sa-key.json"
FOLDER_NAME = "deforest_training"
USER_EMAIL = "farrelafif05@gmail.com"

credentials = service_account.Credentials.from_service_account_file(KEY_PATH, scopes=SCOPES)
service = build("drive", "v3", credentials=credentials)

# Check if folder exists, if not create it
results = service.files().list(q=f"name='{FOLDER_NAME}' and mimeType='application/vnd.google-apps.folder' and 'root' in parents", fields="files(id, name)").execute()
files = results.get("files", [])
if files:
    folder_id = files[0]["id"]
    print(f"Folder exists: {folder_id}")
else:
    metadata = {"name": FOLDER_NAME, "mimeType": "application/vnd.google-apps.folder"}
    folder = service.files().create(body=metadata, fields="id").execute()
    folder_id = folder.get("id")
    print(f"Folder created: {folder_id}")

# Share with user's email
permission = {
    "type": "user",
    "role": "writer",
    "emailAddress": USER_EMAIL
}
result = service.permissions().create(fileId=folder_id, body=permission, sendNotificationEmail=False).execute()
print(f"Shared with {USER_EMAIL}: {result.get('id')}")

link = f"https://drive.google.com/drive/folders/{folder_id}"
print(f"Link: {link}")
