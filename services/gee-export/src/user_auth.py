import json
from pathlib import Path

import ee
import google.auth.transport.requests
from google.oauth2.credentials import Credentials

SCOPES = [
    "openid",
    "email",
    "https://www.googleapis.com/auth/earthengine",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/devstorage.full_control",
]


def _get_token_path():
    return Path(__file__).parents[3] / "config" / "gee_user_credentials.json"


def has_saved_credentials():
    return _get_token_path().exists()


def load_user_credentials() -> Credentials | None:
    token_path = _get_token_path()
    if not token_path.exists():
        return None
    try:
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
        if creds.expired:
            creds.refresh(google.auth.transport.requests.Request())
            _save_credentials(creds)
        return creds
    except Exception as e:
        print(f"[WARN] Gagal load kredensial user: {e}")
        return None


def _save_credentials(creds):
    token_path = _get_token_path()
    data = {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": list(creds.scopes) if creds.scopes else SCOPES,
        "expiry": creds.expiry.isoformat() if creds.expiry else None,
    }
    token_path.parent.mkdir(parents=True, exist_ok=True)
    with open(token_path, "w") as f:
        json.dump(data, f, indent=2)


def init_ee(project: str = None):
    creds = load_user_credentials()
    if creds is None:
        return False
    ee.Initialize(credentials=creds, project=project)
    return True
